#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darwinian Training: component ablation script.

This script runs only Darwin-style sample weighting variants:
1. Full Darwin:        + alpha*confidence - beta*loss - gamma*instability
2. No confidence:      - beta*loss - gamma*instability
3. No loss:            + alpha*confidence - gamma*instability
4. No instability:     + alpha*confidence - beta*loss
5. Confidence only:    + confidence
6. Loss only:          - loss
7. Instability only:   - instability

Supported datasets:
- cifar10
- cifar100
- fashion_mnist

Outputs:
- ablation_summary_<dataset>.csv
- ablation_epoch_<dataset>.csv

The datasets are automatically downloaded by torchvision if not already present.
"""

import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# =========================================================
# Config: edit these values as needed
# =========================================================
DATASET_NAME = "cifar100"          # "cifar10", "cifar100", "fashion_mnist"
SEEDS = [1, 2, 3]  
NOISE_RATES = [0.4, 0.6] 
NOISE_TYPE = "symmetric"

BATCH_SIZE = 128
EPOCHS = 50                    # 50 for CIFAR; 15 for Fashion-MNIST
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
ROOT_DIR = "./data"

# Main Darwin parameters
DARWIN_ALPHA = 2.0
DARWIN_BETA = 0.7
DARWIN_GAMMA = 0.5
DARWIN_EMA_DECAY = 0.9
DARWIN_MIN_WEIGHT = 0.03
DARWIN_WARMUP_EPOCHS = 3
DARWIN_WEIGHT_SHARPNESS = 0.5
DARWIN_NORMALIZE_MEAN = False

PRINT_EVERY_BATCHES = 100
EXCLUSION_THRESHOLD = 0.10

RESULTS_CSV = f"ablation_summary_{DATASET_NAME}.csv"
EPOCH_RESULTS_CSV = f"ablation_epoch_{DATASET_NAME}.csv"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# =========================================================
# Dataset configurations
# =========================================================
DATASET_CONFIGS = {
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "num_classes": 10,
        "in_channels": 3,
        "train_transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ]),
    },
    "cifar100": {
        "dataset_class": datasets.CIFAR100,
        "num_classes": 100,
        "in_channels": 3,
        "train_transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ]),
    },
    "fashion_mnist": {
        "dataset_class": datasets.FashionMNIST,
        "num_classes": 10,
        "in_channels": 1,
        "train_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
    },
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

CFG = DATASET_CONFIGS[DATASET_NAME]


# =========================================================
# Ablation variants
# =========================================================
ABLATION_VARIANTS = {
    "Full Darwin": {
        "use_confidence": True,
        "use_loss": True,
        "use_instability": True,
        "alpha": DARWIN_ALPHA,
        "beta": DARWIN_BETA,
        "gamma": DARWIN_GAMMA,
    },
    "No confidence": {
        "use_confidence": False,
        "use_loss": True,
        "use_instability": True,
        "alpha": DARWIN_ALPHA,
        "beta": DARWIN_BETA,
        "gamma": DARWIN_GAMMA,
    },
    "No loss": {
        "use_confidence": True,
        "use_loss": False,
        "use_instability": True,
        "alpha": DARWIN_ALPHA,
        "beta": DARWIN_BETA,
        "gamma": DARWIN_GAMMA,
    },
    "No instability": {
        "use_confidence": True,
        "use_loss": True,
        "use_instability": False,
        "alpha": DARWIN_ALPHA,
        "beta": DARWIN_BETA,
        "gamma": DARWIN_GAMMA,
    },
    "Confidence only": {
        "use_confidence": True,
        "use_loss": False,
        "use_instability": False,
        "alpha": 1.0,
        "beta": 0.0,
        "gamma": 0.0,
    },
    "Loss only": {
        "use_confidence": False,
        "use_loss": True,
        "use_instability": False,
        "alpha": 0.0,
        "beta": 1.0,
        "gamma": 0.0,
    },
    "Instability only": {
        "use_confidence": False,
        "use_loss": False,
        "use_instability": True,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 1.0,
    },
}


# =========================================================
# Utilities
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def mean_sd(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values.mean(), 0.0
    return values.mean(), values.std(ddof=1)


def print_epoch_result(method, epoch, epochs, train_loss, train_acc, test_loss, test_acc, elapsed, extra_text=""):
    msg = (
        f"[{method}] Epoch {epoch}/{epochs} | "
        f"train loss={train_loss:.4f}, train acc={train_acc:.4f} | "
        f"test loss={test_loss:.4f}, test acc={test_acc:.4f}"
    )
    if extra_text:
        msg += f" | {extra_text}"
    msg += f" | time={format_time(elapsed)}"
    print(msg, flush=True)


# =========================================================
# Dataset wrapper with injected label noise
# =========================================================
class NoisyImageDataset(Dataset):
    def __init__(self, dataset_name, root="./data", train=True, noise_rate=0.0, noise_type="symmetric", download=True):
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]
        dataset_class = self.cfg["dataset_class"]
        transform = self.cfg["train_transform"] if train else self.cfg["test_transform"]

        self.base_dataset = dataset_class(root=root, train=train, download=download, transform=transform)
        self.num_classes = self.cfg["num_classes"]

        targets = self.base_dataset.targets
        self.targets = torch.tensor(targets if isinstance(targets, list) else np.array(targets), dtype=torch.long)
        self.clean_targets = self.targets.clone()
        self.is_noisy = torch.zeros(len(self.targets), dtype=torch.bool)

        if train and noise_rate > 0:
            if noise_type != "symmetric":
                raise ValueError("This ablation script is intended for symmetric noise only.")
            self.inject_symmetric_noise(noise_rate)

    def inject_symmetric_noise(self, noise_rate):
        n_samples = len(self.targets)
        n_noisy = int(noise_rate * n_samples)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        for idx in noisy_indices:
            old_label = int(self.targets[idx].item())
            new_label = random.choice([c for c in range(self.num_classes) if c != old_label])
            self.targets[idx] = new_label
            self.is_noisy[idx] = True
        print(f"Injected symmetric noise into {n_noisy}/{n_samples} samples ({100 * noise_rate:.1f}%).", flush=True)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        y = self.targets[idx]
        clean_y = self.clean_targets[idx]
        noisy_flag = self.is_noisy[idx]
        return x, y, idx, clean_y, noisy_flag


# =========================================================
# Model
# =========================================================
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================================================
# Evaluation and stats
# =========================================================
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for x, y, _, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += len(y)

    return total_loss / total_seen, total_correct / total_seen


def survival_stats(sample_weights, is_noisy, device, exclusion_threshold=0.10):
    weights = sample_weights.detach()
    noisy_mask = is_noisy.to(device)
    clean_mask = ~noisy_mask
    n_total = len(weights)

    return {
        "mean_weight_all": weights.mean().item(),
        "median_weight_all": weights.median().item(),
        "min_weight_all": weights.min().item(),
        "max_weight_all": weights.max().item(),
        "clean_survival": weights[clean_mask].mean().item() if clean_mask.sum().item() > 0 else np.nan,
        "noisy_survival": weights[noisy_mask].mean().item() if noisy_mask.sum().item() > 0 else np.nan,
        "clean_below_threshold": (weights[clean_mask] <= exclusion_threshold).float().mean().item() if clean_mask.sum().item() > 0 else np.nan,
        "noisy_below_threshold": (weights[noisy_mask] <= exclusion_threshold).float().mean().item() if noisy_mask.sum().item() > 0 else np.nan,
        "all_below_threshold": (weights <= exclusion_threshold).float().mean().item(),
        "effective_samples": weights.sum().item(),
        "effective_fraction": weights.sum().item() / n_total,
        "approx_excluded_fraction": 1.0 - (weights.sum().item() / n_total),
    }


# =========================================================
# Ablation trainer
# =========================================================
class AblationFitnessTrainer:
    def __init__(
        self,
        model,
        n_samples,
        variant_name,
        variant_cfg,
        ema_decay=0.9,
        min_weight=0.03,
        warmup_epochs=3,
        weight_sharpness=0.5,
        normalize_mean=False,
        device="cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.variant_name = variant_name
        self.variant_cfg = variant_cfg

        self.alpha = variant_cfg["alpha"]
        self.beta = variant_cfg["beta"]
        self.gamma = variant_cfg["gamma"]
        self.use_confidence = variant_cfg["use_confidence"]
        self.use_loss = variant_cfg["use_loss"]
        self.use_instability = variant_cfg["use_instability"]

        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.warmup_epochs = warmup_epochs
        self.weight_sharpness = weight_sharpness
        self.normalize_mean = normalize_mean

        self.loss_ema = torch.zeros(n_samples, device=device)
        self.conf_ema = torch.zeros(n_samples, device=device)
        self.prev_loss = torch.zeros(n_samples, device=device)
        self.instability_ema = torch.zeros(n_samples, device=device)
        self.sample_weights = torch.ones(n_samples, device=device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _fitness_to_weights(self, fitness):
        weights = torch.sigmoid(self.weight_sharpness * fitness)
        weights = torch.clamp(weights, min=self.min_weight, max=1.0)
        if self.normalize_mean:
            weights = weights / weights.mean().clamp_min(1e-8)
            weights = torch.clamp(weights, min=self.min_weight, max=1.0)
        return weights

    def update_fitness(self, indices, losses, confidences, epoch):
        losses_det = losses.detach()
        conf_det = confidences.detach()
        instability = torch.abs(losses_det - self.prev_loss[indices])
        d = self.ema_decay

        self.loss_ema[indices] = d * self.loss_ema[indices] + (1 - d) * losses_det
        self.conf_ema[indices] = d * self.conf_ema[indices] + (1 - d) * conf_det
        self.instability_ema[indices] = d * self.instability_ema[indices] + (1 - d) * instability
        self.prev_loss[indices] = losses_det

        if epoch <= self.warmup_epochs:
            weights = torch.ones_like(losses_det, device=self.device)
        else:
            fitness = torch.zeros_like(losses_det, device=self.device)
            if self.use_confidence:
                fitness = fitness + self.alpha * self.conf_ema[indices]
            if self.use_loss:
                fitness = fitness - self.beta * self.loss_ema[indices]
            if self.use_instability:
                fitness = fitness - self.gamma * self.instability_ema[indices]
            weights = self._fitness_to_weights(fitness)

        self.sample_weights[indices] = weights

    def train_epoch(self, loader, optimizer, epoch=None):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        n_batches = len(loader)

        for batch_idx, (x, y, idx, _, _) in enumerate(loader, start=1):
            x = x.to(self.device)
            y = y.to(self.device)
            idx = idx.to(self.device)

            logits = self.model(x)
            losses = self.criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            confidences = probs[torch.arange(len(y), device=self.device), y]

            current_weights = self.sample_weights[idx]
            weighted_loss = (current_weights * losses).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            self.update_fitness(idx, losses, confidences, epoch=epoch)

            total_loss += weighted_loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += len(y)

            if batch_idx % PRINT_EVERY_BATCHES == 0 or batch_idx == n_batches:
                print(
                    f"    [{self.variant_name}] epoch {epoch} | batch {batch_idx}/{n_batches} | "
                    f"mean weight={self.sample_weights.mean().item():.4f}",
                    flush=True,
                )

        return total_loss / total_seen, total_correct / total_seen


# =========================================================
# Single run
# =========================================================
def run_single(seed, noise_rate):
    print("\n" + "=" * 90)
    print(f"Dataset={DATASET_NAME} | seed={seed} | noise={noise_rate} | type={NOISE_TYPE}")
    print("=" * 90, flush=True)

    set_seed(seed)

    train_dataset = NoisyImageDataset(
        dataset_name=DATASET_NAME,
        root=ROOT_DIR,
        train=True,
        noise_rate=noise_rate,
        noise_type=NOISE_TYPE,
        download=True,
    )
    test_dataset = NoisyImageDataset(
        dataset_name=DATASET_NAME,
        root=ROOT_DIR,
        train=False,
        noise_rate=0.0,
        noise_type="symmetric",
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    summary_rows = []
    epoch_rows = []

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        print(f"\nStarting ablation variant: {variant_name}", flush=True)

        set_seed(seed)  # same initialization seed for each variant within the same run
        model = SimpleConvNet(in_channels=CFG["in_channels"], num_classes=CFG["num_classes"]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        trainer = AblationFitnessTrainer(
            model=model,
            n_samples=len(train_dataset),
            variant_name=variant_name,
            variant_cfg=variant_cfg,
            ema_decay=DARWIN_EMA_DECAY,
            min_weight=DARWIN_MIN_WEIGHT,
            warmup_epochs=DARWIN_WARMUP_EPOCHS,
            weight_sharpness=DARWIN_WEIGHT_SHARPNESS,
            normalize_mean=DARWIN_NORMALIZE_MEAN,
            device=DEVICE,
        )

        best_test_acc = 0.0
        final_test_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch=epoch)
            test_loss, test_acc = evaluate(model, test_loader, DEVICE)
            best_test_acc = max(best_test_acc, test_acc)
            final_test_acc = test_acc

            stats = survival_stats(
                trainer.sample_weights,
                train_dataset.is_noisy,
                device=DEVICE,
                exclusion_threshold=EXCLUSION_THRESHOLD,
            )

            print_epoch_result(
                variant_name,
                epoch,
                EPOCHS,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                time.time() - t0,
                extra_text=(
                    f"mean weight={stats['mean_weight_all']:.4f} | "
                    f"clean={stats['clean_survival']:.4f} | "
                    f"noisy={stats['noisy_survival']:.4f}"
                ),
            )

            epoch_rows.append({
                "dataset": DATASET_NAME,
                "seed": seed,
                "noise_type": NOISE_TYPE,
                "noise_rate": noise_rate,
                "variant": variant_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "use_confidence": variant_cfg["use_confidence"],
                "use_loss": variant_cfg["use_loss"],
                "use_instability": variant_cfg["use_instability"],
                "alpha": variant_cfg["alpha"],
                "beta": variant_cfg["beta"],
                "gamma": variant_cfg["gamma"],
                **stats,
            })

        final_stats = survival_stats(
            trainer.sample_weights,
            train_dataset.is_noisy,
            device=DEVICE,
            exclusion_threshold=EXCLUSION_THRESHOLD,
        )

        summary_rows.append({
            "dataset": DATASET_NAME,
            "seed": seed,
            "noise_type": NOISE_TYPE,
            "noise_rate": noise_rate,
            "variant": variant_name,
            "best_test_acc": best_test_acc,
            "final_test_acc": final_test_acc,
            "use_confidence": variant_cfg["use_confidence"],
            "use_loss": variant_cfg["use_loss"],
            "use_instability": variant_cfg["use_instability"],
            "alpha": variant_cfg["alpha"],
            "beta": variant_cfg["beta"],
            "gamma": variant_cfg["gamma"],
            "mean_weight_all": final_stats["mean_weight_all"],
            "clean_survival": final_stats["clean_survival"],
            "noisy_survival": final_stats["noisy_survival"],
            "effective_fraction": final_stats["effective_fraction"],
            "approx_excluded_fraction": final_stats["approx_excluded_fraction"],
        })

    return summary_rows, epoch_rows


# =========================================================
# Summary printing
# =========================================================
def print_summary(df):
    print("\n" + "=" * 130)
    print(f"ABLATION SUMMARY ACROSS SEEDS | DATASET={DATASET_NAME}")
    print("=" * 130)

    for noise_rate in sorted(df["noise_rate"].unique()):
        print(f"\nNoise rate = {noise_rate}")
        print("-" * 130)
        print(f"{'Variant':<20} {'Best test acc':<22} {'Final test acc':<22} {'Clean survival':<22} {'Noisy survival':<22}")
        print("-" * 130)
        sub = df[df["noise_rate"] == noise_rate]
        for variant in ABLATION_VARIANTS.keys():
            s = sub[sub["variant"] == variant]
            best_m, best_sd = mean_sd(s["best_test_acc"].tolist())
            final_m, final_sd = mean_sd(s["final_test_acc"].tolist())
            clean_m, clean_sd = mean_sd(s["clean_survival"].tolist())
            noisy_m, noisy_sd = mean_sd(s["noisy_survival"].tolist())
            print(
                f"{variant:<20} "
                f"{best_m:.4f} ± {best_sd:.4f}{'':<8} "
                f"{final_m:.4f} ± {final_sd:.4f}{'':<8} "
                f"{clean_m:.4f} ± {clean_sd:.4f}{'':<8} "
                f"{noisy_m:.4f} ± {noisy_sd:.4f}"
            )
        print("-" * 130)


# =========================================================
# Main
# =========================================================
def main():
    overall_start = time.time()

    print("Using device:", DEVICE, flush=True)
    print(f"DATASET_NAME={DATASET_NAME}", flush=True)
    print(f"SEEDS={SEEDS}", flush=True)
    print(f"NOISE_RATES={NOISE_RATES}", flush=True)
    print(f"EPOCHS={EPOCHS}", flush=True)
    print(f"RESULTS_CSV={RESULTS_CSV}", flush=True)
    print(f"EPOCH_RESULTS_CSV={EPOCH_RESULTS_CSV}", flush=True)
    print("Ablation variants:", list(ABLATION_VARIANTS.keys()), flush=True)

    all_summary_rows = []
    all_epoch_rows = []
    total_runs = len(SEEDS) * len(NOISE_RATES)
    current_run = 0

    for noise_rate in NOISE_RATES:
        for seed in SEEDS:
            current_run += 1
            print(
                f"\n##### Global progress: run {current_run}/{total_runs} "
                f"(dataset={DATASET_NAME}, seed={seed}, noise={noise_rate}) #####",
                flush=True,
            )
            summary_rows, epoch_rows = run_single(seed, noise_rate)
            all_summary_rows.extend(summary_rows)
            all_epoch_rows.extend(epoch_rows)

            pd.DataFrame(all_summary_rows).to_csv(RESULTS_CSV, index=False)
            pd.DataFrame(all_epoch_rows).to_csv(EPOCH_RESULTS_CSV, index=False)
            print(f"Partial summary saved to: {RESULTS_CSV}", flush=True)
            print(f"Partial epoch results saved to: {EPOCH_RESULTS_CSV}", flush=True)

    df_summary = pd.DataFrame(all_summary_rows)
    df_epoch = pd.DataFrame(all_epoch_rows)
    df_summary.to_csv(RESULTS_CSV, index=False)
    df_epoch.to_csv(EPOCH_RESULTS_CSV, index=False)

    print_summary(df_summary)
    print(f"\nSaved summary to: {RESULTS_CSV}", flush=True)
    print(f"Saved epoch results to: {EPOCH_RESULTS_CSV}", flush=True)
    print(f"Total runtime: {format_time(time.time() - overall_start)}", flush=True)


if __name__ == "__main__":
    main()
