import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from utils.mpsl_utils import get_communication_size


class StoreAttentionWrapper(nn.Module):
    """
    Wraps a torchvision nn.MultiheadAttention module to force it to return
    attention weights and stores the class token attention.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.class_token_attention = None

    def forward(self, *args, **kwargs):
        kwargs['need_weights'] = True

        attn_output, attn_weights = self.mha(*args, **kwargs)

        # Extract attention from the class token (index 0) to all other tokens
        # PyTorch MHA with need_weights=True already averages across heads by default
        self.class_token_attention = attn_weights[:, 0, :].detach()

        return attn_output, attn_weights


class AttentionCompressionLayer(nn.Module):
    """
    Compresses tokens and batches based on attention centroids.
    """

    def __init__(self, prev_block: nn.Module, batch_compression: float, token_compression: float):
        super().__init__()
        self.batch_compression = batch_compression
        self.token_compression = token_compression
        self.cluster_ids = None

        # Explicitly store these to sync with label compression later
        self.n_new_batches = 0
        self.n_new_tokens = 0

        # Dynamically wrap the previous block's attention module
        self.attn_wrapper = StoreAttentionWrapper(prev_block.self_attention)
        prev_block.self_attention = self.attn_wrapper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        n_batches, n_tokens, _ = x.size()

        self.n_new_tokens = max(1, int(self.token_compression * n_tokens))
        self.n_new_batches = max(2, int(self.batch_compression * n_batches))
        device = x.device

        class_token_attention = self.attn_wrapper.class_token_attention.to(device)

        # We need some addition to make this compatible with small 
        if self.n_new_batches >= n_batches or n_batches < 2:
            # Bypass K-Means: treat each individual image as its own cluster.
            # This ensures token compression still occurs so sequence lengths match across clients.
            self.n_new_batches = n_batches
            self.cluster_ids = torch.arange(n_batches, device=device)
            centroids = class_token_attention
        else:
            # Standard K-Means clustering
            with torch.no_grad():
                cluster_ids, centroids = kmeans(
                    X=class_token_attention,
                    num_clusters=self.n_new_batches,
                    distance='euclidean',
                    tol=1e-4,
                    iter_limit=500,
                    device=device,
                    tqdm_flag=False
                )
            self.cluster_ids = cluster_ids.detach().to(device)
            centroids = centroids.detach().to(device)

        clustered_activations = []

        for cluster_id in range(self.n_new_batches):
            mask = self.cluster_ids == cluster_id

            if not mask.any():
                continue

            cluster_activations = x[mask]
            average_activation = cluster_activations.mean(dim=0)
            cluster_class_token_attention = centroids[cluster_id, 1:]

            top_k_tokens = torch.topk(
                cluster_class_token_attention,
                k=self.n_new_tokens - 1,
                largest=True,
                sorted=False
            ).indices

            top_k_tokens_indexes = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                top_k_tokens + 1
            ])

            clustered_activations.append(average_activation[top_k_tokens_indexes, :])

        if len(clustered_activations) == 0:
            return x

        return torch.stack(clustered_activations, dim=0)

class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer, batch_compression: float, token_compression: float,
                 num_classes):
        super().__init__()
        self.device = device
        self.split_layer = split_layer

        # 1. Input Processing
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)
        self.image_size = centralized_base_model.vit.image_size

        # 2. Embeddings & Tokens
        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)

        # 3. Transformer Blocks (up to split)
        client_layers = []
        for i in range(self.split_layer):
            client_layers.append(centralized_base_model.vit.encoder.layers[i])
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        # 4. Attach Attention Compression to the last client block
        if self.split_layer > 0:
            last_client_block = self.blocks[-1]
            self.compression_layer = AttentionCompressionLayer(
                prev_block=last_client_block,
                batch_compression=batch_compression,
                token_compression=token_compression
            )
        else:
            self.compression_layer = None

        self.num_classes = num_classes

        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        n = x.shape[0]

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Standard forward pass through client blocks
        x = self.blocks(x)

        return x

    def compress_decompress(self, x):
        if self.compression_layer is not None:
            compressed_x = self.compression_layer(x)
        else:
            compressed_x = x

        # Calculate communication size
        comm_size_bytes = get_communication_size(compressed_x.clone())
        mse = 0.0

        return compressed_x, mse, comm_size_bytes

    def compress_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.compression_layer is None or self.compression_layer.cluster_ids is None:
            return labels

        cluster_ids = self.compression_layer.cluster_ids

        n_new_batches = self.compression_layer.n_new_batches
        new_labels = []

        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        for cluster_id in range(n_new_batches):
            mask = cluster_ids == cluster_id

            if not mask.any():
                continue

            cluster_labels = one_hot_labels[mask]
            cluster_average_label = cluster_labels.mean(dim=0)
            new_labels.append(cluster_average_label)

        if len(new_labels) == 0:
            return one_hot_labels

        return torch.stack(new_labels, dim=0)

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self


class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer):
        super().__init__()
        self.device = device

        # Get all encoder layers from split_layer onwards
        server_layers = list(centralized_base_model.vit.encoder.layers)[split_layer:]
        self.server_blocks = copy.deepcopy(nn.Sequential(*server_layers))

        self.final_layer_norm = copy.deepcopy(centralized_base_model.vit.encoder.ln)
        self.heads = copy.deepcopy(centralized_base_model.vit.heads)

        self.to(device)

    def forward_uncompressed(self, x):
        x = x.to(self.device)

        # Run remaining blocks
        x = self.server_blocks(x)

        # Final Layer Norm and Head
        x = self.final_layer_norm(x)
        x = x[:, 0]
        x = self.heads(x)

        return x

    def forward(self, x):
        return self.forward_uncompressed(x)

    def switch_to_device(self, device):
        self.to(device)
        self.device = device
        return self