# Darwinian Training

Code for **Darwinian Training: History-Dependent Soft Reweighting for Learning with Noisy Labels**.

Darwinian Training is a single-network method for learning with noisy labels. It tracks each training sample's loss, confidence and loss instability across epochs and uses them to set a continuous, strictly positive weight for that sample. It needs no clean validation data and no second network.

## Requirements

```bash
pip install -r requirements.txt
```

The accompanying `requirements.txt` lists the Python dependencies. Use compatible `torch` and `torchvision` builds for your hardware (CPU, CUDA or Apple MPS); MW-Net requires PyTorch 2.0 or later for `torch.func`. The scripts pick the device automatically. CIFAR-10, CIFAR-100 and Fashion-MNIST are downloaded to `./data` on first use.

## Main experiment

```bash
python main_darwinian_training.py
```

The script compares seven methods: Darwinian Training, Co-teaching, GCE, standard training (ERM), an approximately mean-matched random-weight control, TERM and Meta-Weight-Net. It runs symmetric label noise at rates 0.2, 0.4 and 0.6, with seeds 1–5.

Before running, set these two values at the top of the script:

| Dataset | `DATASET_NAME` | `EPOCHS` |
|---|---|---|
| CIFAR-10 | `"cifar10"` | `50` |
| CIFAR-100 | `"cifar100"` | `50` |
| Fashion-MNIST | `"fashion_mnist"` | `15` |

Output files:
- `<dataset>_darwin_results.csv`: final and best test accuracy, plus weight statistics, for each method, seed and noise rate
- `<dataset>_darwin_epoch_results.csv`: the same measurements for every epoch

Protocol notes:
- **Meta-Weight-Net** holds out 1,000 class-balanced training samples with clean labels for meta-validation and trains the classifier on the rest.
- **Co-teaching** uses the true noise rate in its forget-rate schedule and reports accuracy averaged over its two networks.
- **All methods** use the same synthetic label-corruption realisation for each seed and noise rate; MW-Net subsequently excludes its clean meta-validation subset from classifier training.

## Ablation experiment

```bash
python darwinian_training_ablation.py
```

This script trains seven versions of Darwinian Training's fitness score:
- the full score
- the score with confidence, loss or instability removed
- each of those three terms alone

By default it uses CIFAR-100 at noise rates 0.4 and 0.6, with seeds 1–3 and 50 epochs. These settings are at the top of the script.

Output files: `ablation_summary_<dataset>.csv` and `ablation_epoch_<dataset>.csv`.

## Darwinian Training hyperparameters

These values are fixed for all datasets and noise rates:

| Parameter | Value |
|---|---|
| α (confidence) | 2.0 |
| β (loss) | 0.7 |
| γ (instability) | 0.5 |
| Moving-average decay λ | 0.9 |
| Sigmoid sharpness s | 0.5 |
| Minimum weight | 0.03 |
| Warm-up epochs | 3 |

The classifier is trained with Adam: learning rate 1e-3, weight decay 1e-4, batch size 128.

## Reproducibility

The scripts set random seeds and use deterministic cuDNN. Results can still vary slightly across hardware and software versions.

## Citation

If you use this code, please cite the paper.

```bibtex
@misc{keshavarzi2026darwinian,
  title  = {Darwinian Training: History-Dependent Soft Reweighting for Learning with Noisy Labels},
  author = {Keshavarzi, Mahmoud},
  year   = {2026}
}
```
