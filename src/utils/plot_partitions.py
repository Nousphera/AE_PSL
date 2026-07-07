
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datasets import concatenate_datasets
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

def plot():
    fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
            # dataset_kwargs={"cache_dir": os.environ["TORCH_DATA_DIR"]}  # This points to your specific path
        )

    # 1. Figure out how to distribute the partitions
    total_partitions = fds.partitioners["train"].num_partitions

    print("Gathering data for distribution analysis...")
    all_partitions = []
    tenth_partitions = []

    # Load all partitions and create the 10% subsample simultaneously
    for p_id in range(total_partitions):
        partition_hf = fds.load_partition(p_id, "train")
        all_partitions.append(partition_hf)

        # Grab 10% of the rows for this partition
        subsampled = partition_hf.train_test_split(
            train_size=0.1,
            seed=42
        )['train']
        tenth_partitions.append(subsampled)

    # Concatenate to form the full views
    full_dataset = concatenate_datasets(all_partitions)
    tenth_dataset = concatenate_datasets(tenth_partitions)

    # Measure counts for the 'character' label
    full_counts = Counter(full_dataset['character'])
    tenth_counts = Counter(tenth_dataset['character'])

    # Normalize the counts (calculate proportions)
    total_full = sum(full_counts.values())
    total_tenth = sum(tenth_counts.values())

    all_chars = list(set(full_counts.keys()).union(set(tenth_counts.keys())))

    df = pd.DataFrame({
        'Character': all_chars,
        'Full_Norm': [full_counts.get(c, 0) / total_full for c in all_chars],
        'Tenth_Norm': [tenth_counts.get(c, 0) / total_tenth for c in all_chars]
    })

    # Sort by the original distribution for an organized chart
    df = df.sort_values('Full_Norm', ascending=False)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    x = np.arange(len(df['Character']))

    # Plot 1: Full Dataset
    axes[0].bar(x, df['Full_Norm'], color='tab:blue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Character'], rotation=0, fontsize=8)
    axes[0].set_title('Full Dataset (Normalized)')
    axes[0].set_ylabel('Proportion')
    axes[0].set_xlabel('Character Class')

    # Plot 2: 1/10th Dataset (Properly Subsampled)
    axes[1].bar(x, df['Tenth_Norm'], color='tab:orange')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Character'], rotation=0, fontsize=8)
    axes[1].set_title('1/10th Dataset (Normalized - 10% per Partition)')
    axes[1].set_xlabel('Character Class')

    plt.tight_layout()
    plt.show()
    plt.savefig("character_distribution_comparison_new_method.png")

if __name__ == "__main__":
    plot()