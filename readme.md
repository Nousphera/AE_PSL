# AutoEncoder-Compressed Parallel Split Learning for Pre-trained Model Fine-Tuning

**AE-PSL** is a communication-efficient Parallel Split Learning (PSL) framework for distributed fine-tuning of pre-trained foundation models on resource-constrained edge devices.

Instead of relying on heuristic communication compression (e.g., quantization or sparsification), **AE-PSL** uses a lightweight **AutoEncoder (AE)** to compress intermediate activations and gradients exchanged between clients and the server. To ensure compatibility with off-the-shelf pre-trained models, AE-PSL introduces a **two-stage alignment procedure** that adapts the autoencoder to both the pre-trained feature manifold and client-specific feature distributions before distributed fine-tuning.

Extensive experiments on four vision benchmarks demonstrate that AE-PSL achieves a significantly better accuracy–communication trade-off than existing heuristic compression methods.

---

## Paper

**AutoEncoder-Compressed Parallel Split Learning for Pre-trained Model Fine-Tuning**  
*[Bas Meuwissen](www.linkedin.com/in/bas-meuwissen), Vasileios Tsouvalas and Nirvana Meratnia*

*ECML-PKDD 2026, 4th Workshop on Advancements in Federated Learning*

**Paper:**

[Read the pre-print on arXiv](https://arxiv.org/abs/2607.17913)

---

## Installation

Code has been tested with **Python 3.10**.

<details>
<summary><strong>Clone the repository and create a virtual environment</strong></summary>

Clone the repository:

```bash
git clone <repository-url>
cd AE_PSL
```

Create a virtual environment:

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```


</details>


Install the project dependencies:

```bash
pip install -r requirements.txt
```


---

## Datasets

Place all datasets under

```text
./data/torch
```

This is the default location, but another dataset directory can be specified through the experiment configuration.

### Supported datasets

| Dataset | Download |
|---------|----------|
| CIFAR100 | Automatically downloaded via PyTorch |
| Food101 | Automatically downloaded via PyTorch |
| SUN397 | Automatically downloaded via PyTorch |
| FEMNIST | https://huggingface.co/datasets/flwrlabs/femnist |
| ImageNet100 | https://www.kaggle.com/datasets/ambityga/imagenet100 |

### Notes

- **ImageNet100** is used by default during the **AE General Alignment** stage.
- The training **batch size must be divisible by the number of clients**.
- When using **FEMNIST**, local evaluation is automatically enabled. See the paper for details on local versus global evaluation.

---

## Running Experiments

All experiments are launched through

```bash
python orchestrator.py
```

with the desired command-line arguments.

Several example configurations are provided as shell scripts.

### Basic Example

```bash
bash run_basic_example.sh
```

---

## Reproducing Paper Results

#### Main Results (Section 5.3.1)

```bash
bash run_main_results.sh
```

Baseline comparisons:

```bash
bash run_baselines.sh
```

#### Computational Overhead (Section 5.3.2)

Enable FLOP profiling by adding

```bash
--profile_flops true
```

to any experiment.

#### AE Architecture Study (Section 5.3.3)

```bash
bash run_ae_architecture.sh
```

#### Ablation Study (Section 5.3.4)

```bash
bash run_ablation.sh
```

#### No Compression Baseline (R = 1)

```bash
bash run_no_compression.sh
```

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@misc{meuwissen2026autoencodercompressedparallelsplitlearning,
      title={AutoEncoder-Compressed Parallel Split Learning for Pre-trained Model Fine-Tuning}, 
      author={Bas Meuwissen and Vasileios Tsouvalas and Nirvana Meratnia},
      year={2026},
      eprint={2607.17913},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2607.17913}, 
}
```
