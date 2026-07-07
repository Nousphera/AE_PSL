
import torch
from torchvision.transforms import v2
from tqdm import tqdm


from trainers.implementations.experiment_results import ExperimentResults
from trainers.implementations.experiment_trainer import ExperimentTrainer
from utils.fl_utils import fed_avg
from utils.model_utils import get_hidden_dim
from utils.mpsl_utils import get_communication_size, compute_aggregated_loss, bytes_to_megabytes, get_client_name

import torch

from tqdm import tqdm


loss_fn = torch.nn.CrossEntropyLoss()
ae_loss_fn = torch.nn.MSELoss()
import torch.nn.functional as F


def get_gpu_vit_transforms(train: bool):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return v2.Compose([
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(),
            normalize,
        ])
    else:
        return v2.Compose([
            v2.Resize((224, 224), antialias=True),
            normalize,
        ])


class SASLTrainer(ExperimentTrainer):

    def __init__(self):
        super()
        
    def _single_warmup_epoch(self, device, server_model, server_optimizer, global_train_dataloader, client_models: dict, client_model_requires_any_grad, client_optimizers_main: dict, client_optimizers_ae: dict, max_nr_of_batches_in_epoch, epoch_nr, global_args):
        client_ids = range(global_args['nr_of_clients'])

        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {
            _client_id: 0 for _client_id in client_ids}

        use_gpu_transform = global_args['dataset'] == 'cifar100'

        client_batch_mses = {_client_id: [] for _client_id in client_ids}

        for batch_nr, client_batch_dict in enumerate(tqdm(global_train_dataloader)):




            server_optimizer.zero_grad()

            compressed_activations_per_client, client_loss_fn_tuples = dict(), []
            cls_tokens_per_client = dict()

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices, y_per_client = dict(), dict()

            # Iterate ONLY over the clients that have data in this global batch
            for client_id, (X, y) in client_batch_dict.items():
                y = y.to(device)
                X = X.to(device)
                if use_gpu_transform:
                    X = X / 255.0
                    X = get_gpu_vit_transforms(train=True)(X)
                y_per_client[client_id] = y
                nr_of_elements_per_client[client_id] += len(y)

                # FP on client-side model
                # client_optimizers_main[client_id].zero_grad()
                client_optimizers_ae[client_id].zero_grad()

                # General FW pass
                with torch.no_grad():
                    activations = client_models[client_id].forward(X)
                # original_activations_per_client[client_id] = activations.detach().clone()

                # Main Forward Pass (CE Graph)
                # Freeze the encoder so it does NOT record gradients during the main pass
                client_models[client_id].ae_module.freeze(freeze_encoder=False, freeze_decoder=True)

                compressed_activations, cls_tokens, comm_size = client_models[client_id].compress(activations)

                compressed_activations_per_client[client_id] = compressed_activations
                cls_tokens_per_client[client_id] = cls_tokens

                # AE Forward Pass (MSE Graph)
                # 0. Unfreeze the encoder so it can be trained by the MSE loss


                # 2a. Forward pass through the frozen AE decoder on the client-side to  get reconstructions
                # Make sure to detach the intermediate activations so that gradients don't flow back through the client model during the AE loss computation
                reconstructions = client_models[client_id].ae_module.forward(activations.detach())
                # 2b. Compute MSE loss for the reconstructions
                ae_loss = ae_loss_fn(reconstructions, activations.detach())
                # print(f'Warmup -- Client {client_id} AE MSE Batch Loss: {ae_loss.item()}')
                client_batch_mses[client_id].append(ae_loss.item())
                # 2c. Backpropagate through the AE decoder

                ae_loss.backward()

                client_optimizers_ae[client_id].step()

        return client_batch_mses


    def _single_aggregated_warmup_epoch(self, device, server_model, server_optimizer, global_train_dataloader, client_models: dict, client_model_requires_any_grad, client_optimizers_main: dict, client_optimizers_ae: dict, max_nr_of_batches_in_epoch, epoch_nr, global_args):
        client_ids = range(global_args['nr_of_clients'])

        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {
            _client_id: 0 for _client_id in client_ids}

        use_gpu_transform = global_args['dataset'] == 'cifar100'

        client_batch_mses = {_client_id: [] for _client_id in client_ids}

        for batch_nr, client_batch_dict in enumerate(tqdm(global_train_dataloader)):
            server_optimizer.zero_grad()

            compressed_activations_per_client, client_loss_fn_tuples = dict(), []
            cls_tokens_per_client = dict()

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices, y_per_client = dict(), dict()

            # Iterate ONLY over the clients that have data in this global batch
            for client_id, (X, y) in client_batch_dict.items():
                y = y.to(device)
                X = X.to(device)
                if use_gpu_transform:
                    X = X / 255.0
                    X = get_gpu_vit_transforms(train=True)(X)
                y_per_client[client_id] = y
                nr_of_elements_per_client[client_id] += len(y)

                # FP on client-side model
                # client_optimizers_main[client_id].zero_grad()
                if global_args['concurrent_mse_alignment']:
                    client_optimizers_ae[client_id].zero_grad()

                # client_models[client_id].set_lora_trainable(False)
                client_models[client_id].ae_module.freeze(freeze_encoder=False, freeze_decoder=False)

                with torch.no_grad():
                    activations = client_models[client_id].forward(X)
                    # activations = activations.detach().clone().requires_grad_(True)

                # 2a. Forward pass through the frozen AE decoder on the client-side to  get reconstructions
                # Make sure to detach the intermediate activations so that gradients don't flow back through the client model during the AE loss computation
                reconstructions = client_models[client_id].ae_module.forward(activations.detach())
                # 2b. Compute MSE loss for the reconstructions
                ae_loss = ae_loss_fn(reconstructions, activations.detach())
                # print(f'Warmup -- Client {client_id} AE MSE Batch Loss: {ae_loss.item()}')
                client_batch_mses[client_id].append(ae_loss.item())
                # 2c. Backpropagate through the AE decoder

                ae_loss.backward()

                if global_args['concurrent_mse_alignment']:
                    client_optimizers_ae[client_id].step()

        # 1. Calculate weight multipliers based on the number of elements seen by each client
        total_elements = sum(nr_of_elements_per_client.values())
        if total_elements == 0:
            raise ValueError("No data was processed; cannot perform Federated Averaging.")

        client_weight_multipliers = {
            cid: count / total_elements
            for cid, count in nr_of_elements_per_client.items()
        }

        # 2. Extract the ae_module from each client model into a dictionary
        # The fed_avg function expects a dict where keys match the range(nr_of_clients)
        ae_modules_to_aggregate = {
            cid: client_models[cid].ae_module
            for cid in client_ids
        }

        # 3. Perform the aggregation
        # This returns a deep-copied object with averaged parameters
        global_ae_module = fed_avg(
            models=ae_modules_to_aggregate,
            client_weight_multipliers=client_weight_multipliers
        )



        return client_batch_mses, global_ae_module





    def _single_epoch(self, device, server_model, server_optimizer, global_train_dataloader, client_models: dict,
                      client_model_requires_any_grad, client_optimizers_main: dict, client_optimizers_ae: dict,
                      max_nr_of_batches_in_epoch, epoch_nr, global_args):
        client_ids = range(global_args['nr_of_clients'])
        total_server_loss, acc = 0, 0
        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {
            _client_id: 0 for _client_id in client_ids}

        client_model_requires_any_grad = not global_args['freeze_clientside']
        # = Communication tracking =
        client_outgoing_communication_sizes, client_incoming_communication_sizes = {_client_id: 0 for _client_id in
                                                                                    client_ids}, {_client_id: 0 for
                                                                                                  _client_id in
                                                                                                  client_ids}
        server_outgoing_communication_size, server_incoming_communication_size = 0, 0

        client_batch_mses = {_client_id: [] for _client_id in client_ids}
        per_client_server_batch_mses = {_client_id: [] for _client_id in client_ids}

        use_gpu_transform = global_args['dataset'] == 'cifar100'

        hidden_dim = get_hidden_dim(global_args['model'])
        freeze_server_side_ae_decoder = global_args['concurrent_mse_alignment'] or global_args[
            'ae_freeze_decoder_during_finetuning']

        if global_args['concurrent_mse_alignment'] and global_args['ae_freeze_encoder_during_finetuning']:
            raise ValueError(
                "Both 'concurrent_mse_alignment' and 'ae_freeze_encoder_during_finetuning' cannot be true.")

        freeze_client_side_ae_encoder = global_args['ae_freeze_encoder_during_finetuning'] or global_args['freeze_clientside']
        server_model.ae_module.freeze(freeze_encoder=True, freeze_decoder=freeze_server_side_ae_decoder)

        for batch_nr, client_batch_dict in enumerate(tqdm(global_train_dataloader)):
            skip_compression_for_current_batch = (batch_nr < global_args['no_compression_batches'] and epoch_nr <= 1)
            server_optimizer.zero_grad()


            if global_args['small_test_run'] and batch_nr > 3:
                break

            compressed_activations_entire_batch_combined = None
            cls_tokens_entire_batch_combined = None
            original_activations_per_client = dict()
            compressed_activations_per_client, client_loss_fn_tuples = dict(), []
            cls_tokens_per_client = dict()
            mini_batch_indices, y_per_client = dict(), dict()

            for client_id, (X, y) in client_batch_dict.items():
                # Data distribution tracking (Only for first epoch)

                y = y.to(device)
                X = X.to(device)
                if use_gpu_transform:
                    X = X / 255.0
                    X = get_gpu_vit_transforms(train=True)(X)
                y_per_client[client_id] = y
                nr_of_elements_per_client[client_id] += len(y)

                client_optimizers_main[client_id].zero_grad()
                if global_args['concurrent_mse_alignment']:
                    client_optimizers_ae[client_id].zero_grad()

                activations = client_models[client_id].forward(X)
                original_activations_per_client[client_id] = activations.detach().clone()


                client_models[client_id].set_lora_trainable(not global_args['freeze_clientside'])
                client_models[client_id].ae_module.freeze(freeze_encoder=freeze_client_side_ae_encoder,
                                                          freeze_decoder=True)




                if skip_compression_for_current_batch:
                    compressed_activations = activations
                    compressed_activations_per_client[client_id] = compressed_activations
                    # Make sure this has requires_grad=True if it's acting as a dummy
                    cls_tokens = torch.zeros((activations.shape[0], 1, hidden_dim), device=activations.device,
                                             requires_grad=True)
                    client_batch_mses[client_id].append(0.0)
                    comms_size = get_communication_size(activations.clone())
                else:
                    compressed_activations, cls_tokens, comms_size = client_models[client_id].compress(activations)
                    compressed_activations_per_client[client_id] = compressed_activations
                    cls_tokens_per_client[client_id] = cls_tokens

                    reconstructions = client_models[client_id].ae_module.forward(activations.detach())
                    ae_loss = ae_loss_fn(reconstructions, activations.detach())
                    # print(f'Epoch {epoch_nr} Batch {batch_nr} -- Client {client_id} AE MSE Batch Loss: {ae_loss.item()}')
                    client_batch_mses[client_id].append(ae_loss.item())


                    if global_args['concurrent_mse_alignment']:
                        ae_loss.backward()

                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                if compressed_activations_entire_batch_combined is None:
                    compressed_activations_entire_batch_combined = compressed_activations
                    cls_tokens_entire_batch_combined = cls_tokens
                    mini_batch_indices[client_id] = (0, len(compressed_activations))
                else:
                    client_begin_index = len(compressed_activations_entire_batch_combined)
                    mini_batch_indices[client_id] = (client_begin_index, client_begin_index + len(y))
                    compressed_activations_entire_batch_combined = torch.cat(
                        (compressed_activations_entire_batch_combined, compressed_activations))
                    cls_tokens_entire_batch_combined = torch.cat((cls_tokens_entire_batch_combined, cls_tokens))

            # Server-side FP
            final_activations = compressed_activations_entire_batch_combined.detach().clone().requires_grad_(True)
            final_cls_tokens = cls_tokens_entire_batch_combined.detach().clone().requires_grad_(True)

            if skip_compression_for_current_batch:
                predictions = server_model.forward_uncompressed(final_activations)
            else:
                reconstructed_activations = server_model.decompress(final_activations, final_cls_tokens)
                predictions = server_model.forward_uncompressed(reconstructed_activations)

            for client_id in range(global_args['nr_of_clients']):
                if client_id not in mini_batch_indices:
                    continue

                idx_start, idx_end = mini_batch_indices[client_id]

                if not skip_compression_for_current_batch:
                    recon_client = reconstructed_activations[idx_start:idx_end]
                    orig_client = original_activations_per_client[client_id]
                    per_client_server_batch_mses[client_id].append(F.mse_loss(recon_client, orig_client).item())
                else:
                    per_client_server_batch_mses[client_id].append(0.0)

                preds_for_client = predictions[idx_start:idx_end]
                y_for_client = y_per_client[client_id]

                c_loss = loss_fn(preds_for_client, y_for_client)
                client_loss_fn_tuples.append((c_loss, len(y_for_client)))

                # Comms tracking
                client_incoming_communication_sizes[client_id] += get_communication_size(preds_for_client)
                server_outgoing_communication_size += get_communication_size(preds_for_client)
                client_outgoing_communication_sizes[client_id] += get_communication_size(c_loss)
                server_incoming_communication_size += get_communication_size(c_loss)

                # Accuracy (Correcting potential mean-of-means bias)
                y_pred_class = torch.argmax(preds_for_client, dim=1)
                current_batch_acc = (y_pred_class == y_for_client).sum().item() / len(y_for_client)
                acc += current_batch_acc

                c_acc, c_total = client_specific_acc_tuples[client_id]
                client_specific_acc_tuples[client_id] = (c_acc + current_batch_acc, c_total + 1)

            agg_loss = compute_aggregated_loss(client_loss_fn_tuples, len(compressed_activations_entire_batch_combined))
            total_server_loss += agg_loss.item()
            agg_loss.backward()
            server_optimizer.step()

            if client_model_requires_any_grad:
                for client_id in range(global_args['nr_of_clients']):
                    if client_id in mini_batch_indices:
                        start, end = mini_batch_indices[client_id]
                        if skip_compression_for_current_batch:
                            grad_act = final_activations.grad[start:end].clone()
                            torch.autograd.backward(tensors=[compressed_activations_per_client[client_id]],
                                                    grad_tensors=[grad_act])
                            client_incoming_communication_sizes[client_id] += get_communication_size(grad_act)
                            server_outgoing_communication_size += get_communication_size(grad_act)
                        else:
                            if not global_args['compress_cls_token']:
                                grad_act = final_activations.grad[start:end].clone()
                                grad_cls = final_cls_tokens.grad[start:end].clone()
                                torch.autograd.backward(tensors=[compressed_activations_per_client[client_id],
                                                                 cls_tokens_per_client[client_id]],
                                                        grad_tensors=[grad_act, grad_cls])
                            else:
                                grad_act = final_activations.grad[start:end].clone()
                                grad_cls = torch.empty(0, device=grad_act.device)  # Dummy tensor since cls token is not used
                                torch.autograd.backward(tensors=[compressed_activations_per_client[client_id]],
                                                        grad_tensors=[grad_act])
                            client_incoming_communication_sizes[client_id] += (
                                        get_communication_size(grad_act) + get_communication_size(grad_cls))
                            server_outgoing_communication_size += (
                                        get_communication_size(grad_act) + get_communication_size(grad_cls))

                        client_optimizers_main[client_id].step()
                        if global_args['concurrent_mse_alignment']:
                            client_optimizers_ae[client_id].step()



        return total_server_loss, acc, server_incoming_communication_size, server_outgoing_communication_size, \
            client_incoming_communication_sizes, client_outgoing_communication_sizes, client_specific_acc_tuples, \
            nr_of_elements_per_client, client_batch_mses, per_client_server_batch_mses


    def train_epoch(self,
                    device,
                    server_model,
                    server_optimizer,
                    global_train_dataloader,
                    client_models,
                    client_model_requires_any_grad,
                    client_optimizers_main,
                    client_schedulers_main,
                    client_optimizers_ae,
                    client_schedulers_ae,
                    max_nr_of_batches_in_epoch,
                    experiment_results: ExperimentResults,
                    epoch_nr,
                    global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        freeze_client_side_ae_encoder = global_args['ae_freeze_encoder_during_finetuning']
        # for client_model in client_models.values():
        #     # do this before train
        #     client_model.freeze(False)
        #     client_model.ae_module.freeze(freeze_encoder=freeze_client_side_ae_encoder, freeze_decoder=True)

        [model.train() for model in client_models.values()]
        server_model.train()

        total_server_loss, \
        acc, \
        server_incoming_communication_size, \
        server_outgoing_communication_size, \
        client_incoming_communication_sizes, \
        client_outgoing_communication_sizes, \
        client_specific_acc_tuples, \
        nr_of_elements_per_client_dict,\
        client_batch_mses,\
        per_client_server_batch_mses = self._single_epoch(
            device,
            server_model,
            server_optimizer,
            global_train_dataloader,
            client_models,
            client_model_requires_any_grad,
            client_optimizers_main,
            client_optimizers_ae,
            max_nr_of_batches_in_epoch,
            epoch_nr,
            global_args)

        total_server_loss /= max_nr_of_batches_in_epoch
        acc = ((acc / max_nr_of_batches_in_epoch) / global_args['nr_of_clients'])

        experiment_results.add_results(epoch_nr, acc, False)


        if experiment_results is not None:
            for client_id, mse_list in client_batch_mses.items():
                experiment_results.add_client_batch_mse(epoch_nr, client_id, mse_list)
            for client_id, server_mse_list in per_client_server_batch_mses.items():
                experiment_results.add_server_batch_mse(epoch_nr, client_id, server_mse_list)

        print \
            (f'Finished training epoch with server communication overhead: incoming {bytes_to_megabytes(server_incoming_communication_size)} MB & outgoing {bytes_to_megabytes(server_outgoing_communication_size)} MB')

        total_client_outgoing_communication_size, total_client_incoming_communication_size = 0, 0

        for client_id in range(global_args['nr_of_clients']):
            client_specific_total_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]

            # = Communication tracking =
            client_outgoing_comms, client_incoming_comms = client_outgoing_communication_sizes[client_id], client_incoming_communication_sizes[client_id]
            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            print \
                (f'Client-specific train accuracy for {get_client_name(client_id)}: {client_specific_total_acc / client_specific_total_batch_size} with communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB')

            if client_model_requires_any_grad:
                client_schedulers_main[client_id].step()
                if global_args['concurrent_mse_alignment']:
                    client_schedulers_ae[client_id].step()

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes \
            (total_client_incoming_communication_size / global_args['nr_of_clients'])
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes \
            (total_client_outgoing_communication_size / global_args['nr_of_clients'])
        print \
            (f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')
        experiment_results.set_client_communication_overhead(epoch_nr, avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return total_server_loss, acc, nr_of_elements_per_client_dict

    def warmup_epoch(self,
                    device,
                    server_model,
                    server_optimizer,
                    global_train_dataloader,
                    client_models,
                    client_model_requires_any_grad,
                    client_optimizers_main,
                    client_schedulers_main,
                    client_optimizers_ae,
                    client_schedulers_ae,
                    max_nr_of_batches_in_epoch,
                    experiment_results: ExperimentResults,
                    epoch_nr,
                    global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        [model.train() for model in client_models.values()]
        server_model.train()

        client_batch_mses = self._single_warmup_epoch(
            device, server_model, server_optimizer, global_train_dataloader,
            client_models, client_model_requires_any_grad, client_optimizers_main, client_optimizers_ae, max_nr_of_batches_in_epoch, epoch_nr, global_args)

        total_mse = 0

        if experiment_results is not None:
            for client_id, mse_list in client_batch_mses.items():
                experiment_results.add_client_batch_mse(epoch_nr, client_id, mse_list)
                experiment_results.add_server_batch_mse(epoch_nr, client_id, mse_list)

        for client_id, mses in client_batch_mses.items():

            if mses:
                avg_mse = sum(mses) / len(mses)
                print(f"Warmup Epoch -- Client {client_id} Average AE MSE: {avg_mse:.6f}")
                total_mse += avg_mse
        print \
            (f"Warmup Epoch Finished -- Average AE MSE across all clients: {total_mse / global_args['nr_of_clients']:.6f}")

        experiment_results.set_client_communication_overhead(epoch_nr, 0.0,
                                                             0.0)

        return total_mse / global_args['nr_of_clients']

    def client_specific_alignment(self,
                    device,
                    server_model,
                    server_optimizer,
                    global_train_dataloader,
                    client_models,
                    client_model_requires_any_grad,
                    client_optimizers_main,
                    client_schedulers_main,
                    client_optimizers_ae,
                    client_schedulers_ae,
                    max_nr_of_batches_in_epoch,
                    experiment_results: ExperimentResults,
                    epoch_nr,
                    global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        # [model.eval() for model in client_models.values()]
        [model.ae_module.train() for model in client_models.values()]
        # server_model.eval()

        # client_models_copied = {client_id: copy.deepcopy(client_model) for client_id, client_model in client_models.items()}

        client_batch_mses, global_ae_module = self._single_aggregated_warmup_epoch(
                device, server_model, server_optimizer, global_train_dataloader,
                client_models, client_model_requires_any_grad, client_optimizers_main, client_optimizers_ae,
                max_nr_of_batches_in_epoch, epoch_nr, global_args)

        server_model.ae_module.load_state_dict(global_ae_module.state_dict())

        # If we're doing another epoch, we need to update the server-side AE decoders, to align with the next epoch.
        # for client_id in range(global_args['nr_of_clients']):
        #     client_models[client_id].ae_module.patch_decoder.load_state_dict \
        #         (global_ae_module.patch_decoder.state_dict())
        #     # Load individually updated state dict per client
        #     client_models[client_id].ae_module.patch_encoder.load_state_dict(client_models_copied[client_id].ae_module.patch_encoder.state_dict())

        total_mse = 0

        if experiment_results is not None:
            for client_id, mse_list in client_batch_mses.items():
                experiment_results.add_client_batch_mse(epoch_nr, client_id, mse_list)
                experiment_results.add_server_batch_mse(epoch_nr, client_id, mse_list)

        for client_id, mses in client_batch_mses.items():

            if mses:
                avg_mse = sum(mses) / len(mses)
                print(f"Warmup Finished -- Client {client_id} Average AE MSE: {avg_mse:.6f}")
                total_mse += avg_mse
        print(f"Warmup Finished -- Average AE MSE across all clients: {total_mse / global_args['nr_of_clients']:.6f}")

        # in the aggregated warmup, each clients sends its AE the server, which aggregates them and sends them back
        # All AEs are the same
        ae_comm_size = client_models[0].ae_module.get_decoder_communication_size()

        comm_overhead = ae_comm_size

        experiment_results.set_client_communication_overhead(epoch_nr, comm_overhead,
                                                             comm_overhead)

        return total_mse / global_args['nr_of_clients']

    def test_epoch_global_evaluation(self, client_model, server_model, dataloader, device, experiment_results: ExperimentResults, epoch_nr, use_gpu_transform, compression_during_global_eval):
        client_model.eval()
        server_model.eval()

        total_server_loss, acc = 0, 0

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)

        with torch.no_grad():
            for batch_nr in tqdm(range(nr_of_batches)):
                try:
                    X, y = next(data_iter)
                except StopIteration:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    continue
                X = X.to(device)

                if use_gpu_transform:
                    X = X.to(device) / 255.0  # Make sure X goes to device and is normalized to [0, 1] range
                    X = get_gpu_vit_transforms(train=False)(X)

                y = y.to(device)

                # FP on client-side and model followed by FP on server-side model
                if compression_during_global_eval:
                    client_out = client_model.forward(X)
                    intermediate, cls_tokens, _ = client_model.compress(client_out)
                    decompressed = server_model.decompress(intermediate, cls_tokens)
                    predictions = server_model.forward_uncompressed(decompressed)
                else:
                    X = client_model.forward(X)
                    predictions = server_model.forward_uncompressed(X)


                # Loss computation on client-side
                client_loss = loss_fn(predictions, y)

                # Metrics
                y_pred_class = torch.argmax(predictions, dim=1)
                current_batch_acc = (y_pred_class == y).sum().item() / len(y)
                acc += current_batch_acc

                # BP on server-side
                batch_size = len(y)
                agg_loss = compute_aggregated_loss([(client_loss, batch_size)], batch_size)

                total_server_loss += agg_loss.item()

        total_server_loss /= nr_of_batches
        acc = acc / nr_of_batches

        if experiment_results is not None: experiment_results.add_results(epoch_nr, acc, True)

        return total_server_loss, acc

    def test_epoch_distributed(self, client_models: dict, server_model, dataloaders: dict, device,
                               experiment_results: ExperimentResults, epoch_nr,
                               use_gpu_transform, compression_during_local_eval):
        for client_model in client_models.values():
            client_model.eval()
        server_model.eval()

        # Trackers for global aggregation
        global_total_samples = 0
        global_total_correct = 0.0
        global_total_loss = 0.0

        # Trackers for per-client reporting
        client_results = {}

        with torch.no_grad():
            for client_id, client_model in client_models.items():
                dataloader = dataloaders[client_id]
                client_total_loss = 0.0
                client_total_correct = 0
                client_samples = 0

                for X, y in tqdm(dataloader, desc=f"Client {client_id}"):
                    X, y = X.to(device), y.to(device)

                    if use_gpu_transform:
                        X = X / 255.0
                        X = get_gpu_vit_transforms(train=False)(X)

                    if compression_during_local_eval:
                        client_out = client_model.forward(X)
                        intermediate, cls_tokens, _ = client_model.compress(client_out)
                        decompressed = server_model.decompress(intermediate, cls_tokens)
                        predictions = server_model.forward_uncompressed(decompressed)
                    else:
                        client_out = client_model.forward(X)
                        predictions = server_model.forward_uncompressed(client_out)


                    # Compute Loss and Accuracy for this batch
                    batch_loss = loss_fn(predictions, y)

                    batch_size = y.size(0)
                    y_pred_class = torch.argmax(predictions, dim=1)
                    batch_correct = (y_pred_class == y).sum().item()

                    # Accumulate Client Metrics
                    client_total_loss += batch_loss.item() * batch_size
                    client_total_correct += batch_correct
                    client_samples += batch_size

                # Compute Per-Client Final Metrics
                client_avg_loss = client_total_loss / client_samples if client_samples > 0 else 0
                client_avg_acc = client_total_correct / client_samples if client_samples > 0 else 0

                client_results[client_id] = {
                    "loss": client_avg_loss,
                    "acc": client_avg_acc,
                    "samples": client_samples
                }

                # Accumulate Global Metrics
                global_total_loss += client_total_loss
                global_total_correct += client_total_correct
                global_total_samples += client_samples

        # Final Global Calculations
        avg_global_loss = global_total_loss / global_total_samples if global_total_samples > 0 else 0
        avg_global_acc = global_total_correct / global_total_samples if global_total_samples > 0 else 0

        if experiment_results is not None:
            experiment_results.add_results(epoch_nr, avg_global_acc, True)
            experiment_results.distributed_eval[epoch_nr] = client_results

        return avg_global_loss, avg_global_acc, client_results

