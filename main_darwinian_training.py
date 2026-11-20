"""
Darwinian Training: Soft Fitness-Based Reweighting for Noisy Labels

Datasets: CIFAR-10, CIFAR-100, Fashion-MNIST.
Methods: Darwinian Training, Co-teaching, GCE, standard training,
random-weight control, TERM, and Meta-Weight-Net (MW-Net).

For each dataset, the script evaluates the specified noise rates and seeds
and saves per-run and per-epoch results as CSV files.
"""

import time
import contextlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.func import functional_call
from torchvision import datasets, transforms


# =========================================================
# Config
# =========================================================
DATASET_NAME = "cifar10"   # "cifar10", "cifar100", "fashion_mnist"

SEEDS = [1, 2, 3, 4, 5]
NOISE_RATES = [0.2, 0.4, 0.6]
NOISE_TYPES = ["symmetric"]

BATCH_SIZE = 128
EPOCHS = 50  # 50 for "cifar10" and "cifar100", 15 for "fashion_mnist"
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

# Darwin parameters
DARWIN_ALPHA = 2.0
DARWIN_BETA = 0.7
DARWIN_GAMMA = 0.5
DARWIN_EMA_DECAY = 0.9
DARWIN_MIN_WEIGHT = 0.03
DARWIN_WARMUP_EPOCHS = 3
DARWIN_WEIGHT_SHARPNESS = 0.5
DARWIN_NORMALIZE_MEAN = False

# GCE parameters
GCE_Q = 0.7

# Co-teaching parameters
COTEACH_NUM_GRADUAL = 10
COTEACH_EXPONENT = 1.0

# TERM parameters
TERM_T = -0.25

# Meta-Weight-Net parameters
META_SIZE = 1000
META_BATCH_SIZE = 128
MW_HIDDEN_SIZE = 100
MW_META_LR = 1e-3
MW_INNER_LR = LR
MW_EPS = 1e-8

# Statistics / reporting
PRINT_EVERY_BATCHES = 100
EXCLUSION_THRESHOLD = 0.10

ROOT_DIR = "./data"
RESULTS_CSV = f"{DATASET_NAME}_darwin_results.csv"
EPOCH_RESULTS_CSV = f"{DATASET_NAME}_darwin_epoch_results.csv"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

DEVICE = torch.device(device)


# =========================================================
# Dataset configurations
# =========================================================
DATASET_CONFIGS = {
    "cifar10": {
        "dataset_class": datasets.CIFAR10,
        "num_classes": 10,
        "in_channels": 3,
        "image_size": 32,
        "train_transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    },

    "cifar100": {
        "dataset_class": datasets.CIFAR100,
        "num_classes": 100,
        "in_channels": 3,
        "image_size": 32,
        "train_transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
    },

    "fashion_mnist": {
        "dataset_class": datasets.FashionMNIST,
        "num_classes": 10,
        "in_channels": 1,
        "image_size": 32,
        "train_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    }
}

if DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

CFG = DATASET_CONFIGS[DATASET_NAME]

# Clean meta-validation images use deterministic evaluation preprocessing.
for _dataset_cfg in DATASET_CONFIGS.values():
    _dataset_cfg["meta_transform"] = _dataset_cfg["test_transform"]

# =========================================================
# CIFAR download helper
# =========================================================
# This helper is used only for CIFAR-10/CIFAR-100. It keeps the
# torchvision dataset format unchanged, but avoids the common HTTP 503
# failure that can occur when torchvision checks the Toronto server.

import os
import tarfile
import urllib.request
import shutil

CIFAR_ARCHIVES = {
    "cifar10": {
        "filename": "cifar-10-python.tar.gz",
        "folder": "cifar-10-batches-py",
        "urls": [
            "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
            "https://storage.googleapis.com/cvdf-datasets/cifar/cifar-10-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        ],
    },
    "cifar100": {
        "filename": "cifar-100-python.tar.gz",
        "folder": "cifar-100-python",
        "urls": [
            "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz",
            "https://storage.googleapis.com/cvdf-datasets/cifar/cifar-100-python.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        ],
    },
}


def _download_file(url, destination):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def ensure_cifar_available(dataset_name, root):
    """Download/extract CIFAR manually if torchvision cannot access it."""
    if dataset_name not in CIFAR_ARCHIVES:
        return

    info = CIFAR_ARCHIVES[dataset_name]
    os.makedirs(root, exist_ok=True)

    expected_folder = os.path.join(root, info["folder"])
    archive_path = os.path.join(root, info["filename"])

    if os.path.isdir(expected_folder):
        return

    if not os.path.isfile(archive_path):
        last_error = None
        for url in info["urls"]:
            try:
                print(f"Downloading {dataset_name} from: {url}", flush=True)
                _download_file(url, archive_path)
                break
            except Exception as err:
                last_error = err
                print(f"Download failed from: {url}", flush=True)
                if os.path.isfile(archive_path):
                    try:
                        os.remove(archive_path)
                    except OSError:
                        pass
        else:
            raise RuntimeError(
                f"Could not download {dataset_name}. Please manually download {info['filename']} "
                f"and place it in {os.path.abspath(root)}, then run the script again. "
                f"Last error: {last_error}"
            )

    print(f"Extracting {archive_path}...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)


def load_dataset_safely(dataset_name, dataset_class, root, train, download, transform):
    if dataset_name in CIFAR_ARCHIVES and download:
        ensure_cifar_available(dataset_name, root)
        download = False

    return dataset_class(
        root=root,
        train=train,
        download=download,
        transform=transform
    )

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


def mean_sd(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values.mean(), 0.0
    return values.mean(), values.std(ddof=1)


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


def make_forget_rate_schedule(noise_rate, epochs, num_gradual=10, exponent=1.0):
    schedule = np.ones(epochs) * noise_rate
    gradual = np.linspace(0, noise_rate ** exponent, num_gradual)
    schedule[:num_gradual] = gradual
    return schedule


# =========================================================
# Dataset wrapper with injected label noise
# =========================================================
class NoisyImageDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        root="./data",
        train=True,
        noise_rate=0.0,
        noise_type="symmetric",
        download=True
    ):
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]

        dataset_class = self.cfg["dataset_class"]
        transform = self.cfg["train_transform"] if train else self.cfg["test_transform"]

        self.base_dataset = load_dataset_safely(
            dataset_name=dataset_name,
            dataset_class=dataset_class,
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        if not hasattr(self.base_dataset, "targets"):
            raise ValueError(f"Dataset {dataset_name} does not expose .targets")

        self.num_classes = self.cfg["num_classes"]
        self.noise_type = noise_type

        targets = self.base_dataset.targets
        if isinstance(targets, list):
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = torch.tensor(np.array(targets), dtype=torch.long)

        self.clean_targets = self.targets.clone()
        self.is_noisy = torch.zeros(len(self.targets), dtype=torch.bool)

        if train and noise_rate > 0:
            self.inject_noise(noise_rate, noise_type)

    def inject_noise(self, noise_rate, noise_type="symmetric"):
        if noise_type != "symmetric":
            raise ValueError(f"Unknown noise_type: {noise_type}")

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

        print(
            f"Injected symmetric noise into {n_noisy}/{n_samples} samples "
            f"({100 * noise_rate:.1f}%).",
            flush=True
        )

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
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =========================================================
# Losses
# =========================================================
class GCELoss(nn.Module):
    def __init__(self, q=0.7, eps=1e-8):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        p = probs[torch.arange(len(targets), device=logits.device), targets]
        p = torch.clamp(p, min=self.eps, max=1.0)
        loss = (1.0 - p.pow(self.q)) / self.q
        return loss.mean()


# =========================================================
# Evaluation
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


def evaluate_coteaching(model1, model2, loader, device):
    loss1, acc1 = evaluate(model1, loader, device)
    loss2, acc2 = evaluate(model2, loader, device)
    return 0.5 * (loss1 + loss2), 0.5 * (acc1 + acc2)


# =========================================================
# Standard training
# =========================================================
def train_standard_epoch(model, loader, optimizer, device, epoch=None, method_name="Standard"):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    n_batches = len(loader)

    for batch_idx, (x, y, _, _, _) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += len(y)

        if batch_idx % PRINT_EVERY_BATCHES == 0 or batch_idx == n_batches:
            print(f"    [{method_name}] epoch {epoch} | batch {batch_idx}/{n_batches}", flush=True)

    return total_loss / total_seen, total_correct / total_seen


# =========================================================
# GCE training
# =========================================================
def train_gce_epoch(model, loader, optimizer, device, q=0.7, epoch=None, method_name="GCE"):
    model.train()
    criterion = GCELoss(q=q)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    n_batches = len(loader)

    for batch_idx, (x, y, _, _, _) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += len(y)

        if batch_idx % PRINT_EVERY_BATCHES == 0 or batch_idx == n_batches:
            print(f"    [{method_name}] epoch {epoch} | batch {batch_idx}/{n_batches}", flush=True)

    return total_loss / total_seen, total_correct / total_seen


# =========================================================
# Co-teaching training
# =========================================================
def train_coteaching_epoch(
    model1, model2, loader, optimizer1, optimizer2, device,
    forget_rate, epoch=None, method_name="Co-teaching"
):
    model1.train()
    model2.train()

    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total_correct_1 = 0
    total_correct_2 = 0
    total_seen = 0

    n_batches = len(loader)

    for batch_idx, (x, y, _, _, _) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        logits1 = model1(x)
        logits2 = model2(x)

        losses1 = criterion(logits1, y)
        losses2 = criterion(logits2, y)

        remember_rate = 1.0 - forget_rate
        n_remember = max(1, int(remember_rate * len(y)))

        ind1_sorted = torch.argsort(losses1)
        ind2_sorted = torch.argsort(losses2)

        ind1_update = ind2_sorted[:n_remember]
        ind2_update = ind1_sorted[:n_remember]

        loss1 = losses1[ind1_update].mean()
        loss2 = losses2[ind2_update].mean()

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        batch_loss = 0.5 * (loss1.item() + loss2.item())
        total_loss += batch_loss * len(y)
        total_correct_1 += (logits1.argmax(dim=1) == y).sum().item()
        total_correct_2 += (logits2.argmax(dim=1) == y).sum().item()
        total_seen += len(y)

        if batch_idx % PRINT_EVERY_BATCHES == 0 or batch_idx == n_batches:
            print(
                f"    [{method_name}] epoch {epoch} | batch {batch_idx}/{n_batches} | "
                f"forget_rate={forget_rate:.4f} | remember_rate={remember_rate:.4f}",
                flush=True
            )

    avg_acc = 0.5 * ((total_correct_1 / total_seen) + (total_correct_2 / total_seen))
    return total_loss / total_seen, avg_acc


# =========================================================
# Darwinian adaptive weighting trainer
# =========================================================
class FitnessTrainer:
    def __init__(
        self,
        model,
        n_samples,
        alpha=2.0,
        beta=1.0,
        gamma=1.0,
        ema_decay=0.9,
        min_weight=0.05,
        warmup_epochs=1,
        weight_sharpness=1.0,
        normalize_mean=False,
        device="cpu"
    ):
        self.model = model.to(device)
        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
        fitness = self.weight_sharpness * fitness
        weights = torch.sigmoid(fitness)
        weights = torch.clamp(weights, min=self.min_weight, max=1.0)

        if self.normalize_mean:
            mean_w = weights.mean().clamp_min(1e-8)
            weights = weights / mean_w
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
            fitness = (
                self.alpha * self.conf_ema[indices]
                - self.beta * self.loss_ema[indices]
                - self.gamma * self.instability_ema[indices]
            )
            weights = self._fitness_to_weights(fitness)

        self.sample_weights[indices] = weights

    def train_epoch(self, loader, optimizer, epoch=None, method_name="Darwin"):
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
                mean_weight = self.sample_weights.mean().item()
                effective_fraction = mean_weight
                print(
                    f"    [{method_name}] epoch {epoch} | batch {batch_idx}/{n_batches} | "
                    f"mean weight={mean_weight:.4f} | effective fraction≈{effective_fraction:.4f}",
                    flush=True
                )

        return total_loss / total_seen, total_correct / total_seen


# =========================================================
# Random-weight control
# =========================================================
class RandomWeightTrainer:
    def __init__(self, model, n_samples, min_weight=0.05, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.min_weight = min_weight
        self.sample_weights = torch.ones(n_samples, device=device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def refresh_weights(self, target_mean_weight=0.4):
        weights = torch.rand_like(self.sample_weights)
        weights = self.min_weight + (1.0 - self.min_weight) * weights

        current_mean = weights.mean().clamp_min(1e-8)
        weights = weights * (target_mean_weight / current_mean)
        weights = torch.clamp(weights, min=self.min_weight, max=1.0)

        self.sample_weights = weights

    def train_epoch(self, loader, optimizer, target_mean_weight=0.4, epoch=None, method_name="Random control"):
        self.refresh_weights(target_mean_weight=target_mean_weight)

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

            weights = self.sample_weights[idx]
            weighted_loss = (weights * losses).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += len(y)

            if batch_idx % PRINT_EVERY_BATCHES == 0 or batch_idx == n_batches:
                mean_weight = self.sample_weights.mean().item()
                effective_fraction = mean_weight
                print(
                    f"    [{method_name}] epoch {epoch} | batch {batch_idx}/{n_batches} | "
                    f"mean weight={mean_weight:.4f} | effective fraction≈{effective_fraction:.4f}",
                    flush=True
                )

        return total_loss / total_seen, total_correct / total_seen


# =========================================================
# Weight / survival stats
# =========================================================
def survival_stats(sample_weights, is_noisy, device, exclusion_threshold=0.10):
    weights = sample_weights.detach()
    noisy_mask = is_noisy.to(device)
    clean_mask = ~noisy_mask

    n_total = len(weights)
    n_clean = clean_mask.sum().item()
    n_noisy = noisy_mask.sum().item()

    mean_weight_all = weights.mean().item()
    median_weight_all = weights.median().item()
    min_weight_all = weights.min().item()
    max_weight_all = weights.max().item()

    clean_survival = weights[clean_mask].mean().item() if n_clean > 0 else np.nan
    noisy_survival = weights[noisy_mask].mean().item() if n_noisy > 0 else np.nan

    clean_below_thr = (weights[clean_mask] <= exclusion_threshold).float().mean().item() if n_clean > 0 else np.nan
    noisy_below_thr = (weights[noisy_mask] <= exclusion_threshold).float().mean().item() if n_noisy > 0 else np.nan
    all_below_thr = (weights <= exclusion_threshold).float().mean().item()

    effective_samples = weights.sum().item()
    effective_fraction = effective_samples / n_total
    approx_excluded_fraction = 1.0 - effective_fraction

    return {
        "mean_weight_all": mean_weight_all,
        "median_weight_all": median_weight_all,
        "min_weight_all": min_weight_all,
        "max_weight_all": max_weight_all,
        "clean_survival": clean_survival,
        "noisy_survival": noisy_survival,
        "clean_below_threshold": clean_below_thr,
        "noisy_below_threshold": noisy_below_thr,
        "all_below_threshold": all_below_thr,
        "effective_samples": effective_samples,
        "effective_fraction": effective_fraction,
        "approx_excluded_fraction": approx_excluded_fraction,
    }


def load_base_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    transform,
):
    cfg = DATASET_CONFIGS[dataset_name]
    dataset_class = cfg["dataset_class"]

    if dataset_name in CIFAR_ARCHIVES:
        ensure_cifar_available(dataset_name, root)
        download = False
    else:
        download = True

    return dataset_class(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


# =========================================================
# TERM and Meta-Weight-Net data and training
# =========================================================

def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


class IndexedDataset(Dataset):
    """A subset view that can use clean or corrupted labels."""

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        labels: torch.Tensor,
        clean_labels: torch.Tensor,
        noisy_mask: torch.Tensor,
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels.long()
        self.clean_labels = clean_labels.long()
        self.noisy_mask = noisy_mask.bool()

        if not (
            len(self.indices)
            == len(self.labels)
            == len(self.clean_labels)
            == len(self.noisy_mask)
        ):
            raise ValueError("Dataset arrays must have equal lengths.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_idx: int):
        original_idx = int(self.indices[local_idx])
        x, _ = self.base_dataset[original_idx]
        return (
            x,
            self.labels[local_idx],
            local_idx,
            self.clean_labels[local_idx],
            self.noisy_mask[local_idx],
        )


def get_targets(base_dataset) -> torch.Tensor:
    targets = base_dataset.targets
    if isinstance(targets, list):
        return torch.tensor(targets, dtype=torch.long)
    return torch.tensor(np.asarray(targets), dtype=torch.long)


def stratified_meta_split(
    targets: torch.Tensor,
    meta_size: int,
    num_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an approximately class-balanced clean meta split."""
    rng = np.random.default_rng(seed)
    targets_np = targets.numpy()

    per_class = meta_size // num_classes
    remainder = meta_size % num_classes
    meta_indices: List[int] = []

    for cls in range(num_classes):
        cls_idx = np.flatnonzero(targets_np == cls)
        rng.shuffle(cls_idx)
        take = per_class + (1 if cls < remainder else 0)
        if take > len(cls_idx):
            raise ValueError(f"Not enough examples in class {cls} for meta split.")
        meta_indices.extend(cls_idx[:take].tolist())

    meta_indices = np.asarray(meta_indices, dtype=np.int64)
    rng.shuffle(meta_indices)

    all_indices = np.arange(len(targets_np), dtype=np.int64)
    train_indices = np.setdiff1d(all_indices, meta_indices, assume_unique=False)
    return train_indices, meta_indices


def _set_seed_like_original(seed: int) -> None:
    """Match the global RNG setup used by the main training script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def replicate_original_noise(
    clean_targets: torch.Tensor,
    noise_rate: float,
    num_classes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduce the exact symmetric-noise call sequence used in the main training script.

    This must be applied to the complete training-label vector in its original
    order before MW-Net's clean meta split is created.
    """
    _set_seed_like_original(seed)

    targets = clean_targets.clone()
    is_noisy = torch.zeros(len(targets), dtype=torch.bool)
    n_noisy = int(noise_rate * len(targets))
    noisy_indices = np.random.choice(len(targets), n_noisy, replace=False)

    for idx in noisy_indices:
        old_label = int(targets[idx].item())
        new_label = random.choice(
            [c for c in range(num_classes) if c != old_label]
        )
        targets[idx] = new_label
        is_noisy[idx] = True

    return targets, is_noisy


@dataclass
class DataBundle:
    train_dataset: IndexedDataset
    meta_dataset: IndexedDataset
    test_dataset: IndexedDataset
    train_loader: DataLoader
    meta_loader: Optional[DataLoader]
    test_loader: DataLoader


def build_data(
    dataset_name: str,
    noise_rate: float,
    seed: int,
    needs_meta: bool,
) -> DataBundle:
    """
    Construct method-appropriate data while sharing the original noise draw.

    TERM uses all 50,000 noisy training observations. MW-Net uses 49,000 noisy
    training observations plus a disjoint, class-balanced clean meta set of
    1,000 observations. Noise is first generated on all 50,000 observations so
    that its realization matches the main training script exactly.
    """
    cfg = DATASET_CONFIGS[dataset_name]

    train_base = load_base_dataset(
        dataset_name, ROOT_DIR, train=True, transform=cfg["train_transform"]
    )
    meta_base = load_base_dataset(
        dataset_name, ROOT_DIR, train=True, transform=cfg["meta_transform"]
    )
    test_base = load_base_dataset(
        dataset_name, ROOT_DIR, train=False, transform=cfg["test_transform"]
    )

    full_clean_targets = get_targets(train_base)
    test_targets = get_targets(test_base)

    full_noisy_targets, full_noisy_mask = replicate_original_noise(
        full_clean_targets,
        noise_rate=noise_rate,
        num_classes=cfg["num_classes"],
        seed=seed,
    )

    if needs_meta:
        train_indices, meta_indices = stratified_meta_split(
            full_clean_targets,
            meta_size=META_SIZE,
            num_classes=cfg["num_classes"],
            seed=seed,
        )
    else:
        train_indices = np.arange(len(full_clean_targets), dtype=np.int64)
        meta_indices = np.empty(0, dtype=np.int64)

    train_dataset = IndexedDataset(
        train_base,
        train_indices,
        full_noisy_targets[train_indices],
        full_clean_targets[train_indices],
        full_noisy_mask[train_indices],
    )

    # The meta set always retains clean labels, including observations that the
    # full-set corruption draw would otherwise have relabelled.
    meta_dataset = IndexedDataset(
        meta_base,
        meta_indices,
        full_clean_targets[meta_indices],
        full_clean_targets[meta_indices],
        torch.zeros(len(meta_indices), dtype=torch.bool),
    )

    test_indices = np.arange(len(test_targets), dtype=np.int64)
    test_dataset = IndexedDataset(
        test_base,
        test_indices,
        test_targets,
        test_targets,
        torch.zeros(len(test_targets), dtype=torch.bool),
    )

    # Keep loader RNG streams independent. Iterating through MW-Net's meta
    # loader therefore cannot alter the future training-loader shuffle order.
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    meta_generator = torch.Generator()
    meta_generator.manual_seed(seed + 1_000_000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )

    meta_loader: Optional[DataLoader]
    if needs_meta:
        meta_loader = DataLoader(
            meta_dataset,
            batch_size=META_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            generator=meta_generator,
        )
    else:
        meta_loader = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
    )

    print(
        f"{dataset_name}: train={len(train_dataset)} "
        f"(corrupted={int(train_dataset.noisy_mask.sum())}), "
        f"meta={len(meta_dataset)}, test={len(test_dataset)}",
        flush=True,
    )

    return DataBundle(
        train_dataset=train_dataset,
        meta_dataset=meta_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        meta_loader=meta_loader,
        test_loader=test_loader,
    )


class DifferentiableMaxPool2d(nn.Module):
    """
    MaxPool2d wrapper compatible with MW-Net's higher-order gradients.

    PyTorch's max_pool2d double backward requires argmax indices from the
    forward pass. Returning only the pooled values keeps the same model
    interface and numerical output as nn.MaxPool2d.
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        pooled, _indices = nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )
        return pooled


def term_objective(losses: torch.Tensor, tilt: float) -> torch.Tensor:
    """
    Numerically stable mini-batch TERM objective:
        (1/t) log(mean(exp(t * loss_i)))

    As t -> 0 this approaches mean(loss). A negative t emphasizes lower-loss
    observations and is the robust-to-outliers setting.
    """
    if abs(tilt) < 1e-12:
        return losses.mean()

    z = tilt * losses
    return (torch.logsumexp(z, dim=0) - math.log(losses.numel())) / tilt


def train_term_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    tilt: float,
    epoch: int,
) -> Tuple[float, float, float, float]:
    model.train()

    total_objective = 0.0
    total_ce = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (x, y, _, _, _) in enumerate(loader, start=1):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        per_sample_ce = nn.functional.cross_entropy(logits, y, reduction="none")
        objective = term_objective(per_sample_ce, tilt)
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            n = y.numel()
            total_objective += objective.item() * n
            total_ce += per_sample_ce.sum().item()
            total_correct += (logits.argmax(1) == y).sum().item()
            total_seen += n

        if PRINT_EVERY_BATCHES and batch_idx % PRINT_EVERY_BATCHES == 0:
            print(
                f"[TERM] epoch={epoch} batch={batch_idx}/{len(loader)} "
                f"objective={objective.item():.4f}",
                flush=True,
            )

    return (
        total_objective / total_seen,
        total_ce / total_seen,
        total_correct / total_seen,
        float(total_seen),
    )


class MetaWeightNet(nn.Module):
    """One-hidden-layer mapping from scalar training loss to nonnegative weight."""

    def __init__(self, hidden_size: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        if loss_values.ndim == 1:
            loss_values = loss_values.unsqueeze(1)
        return self.net(loss_values)


@contextlib.contextmanager
def bn_stats_frozen(model: nn.Module):
    """Freeze BatchNorm running-statistic updates and restore module flags."""
    bn_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    previous_flags = [module.training for module in bn_modules]
    try:
        for module in bn_modules:
            module.eval()
        yield
    finally:
        for module, was_training in zip(bn_modules, previous_flags):
            module.train(was_training)


def train_mwnet_epoch(
    model: nn.Module,
    vnet: MetaWeightNet,
    train_loader: DataLoader,
    meta_loader: DataLoader,
    model_optimizer: optim.Optimizer,
    vnet_optimizer: optim.Optimizer,
    epoch: int,
) -> Tuple[float, float, float, float, float]:
    """
    Perform one epoch of one-step bilevel MW-Net training.

    BatchNorm running statistics are frozen during both virtual and meta
    forwards and updated only once per batch during the real classifier step.
    The functional-call buffers are cloned to prevent in-place writes from
    reaching the live classifier.
    """
    model.train()
    vnet.train()
    meta_iter = cycle_loader(meta_loader)

    total_weighted_loss = 0.0
    total_unweighted_ce = 0.0
    total_correct = 0
    total_seen = 0
    total_weight = 0.0

    for batch_idx, train_batch in enumerate(train_loader, start=1):
        x_train, y_train, _, _, _ = train_batch
        x_meta, y_meta, _, _, _ = next(meta_iter)

        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        x_meta = x_meta.to(DEVICE)
        y_meta = y_meta.to(DEVICE)

        # A. Update the weighting network through a virtual classifier step.
        with bn_stats_frozen(model):
            params = OrderedDict(model.named_parameters())
            buffers = OrderedDict(
                (name, buffer.clone())
                for name, buffer in model.named_buffers()
            )

            virtual_logits = model(x_train)
            virtual_losses = nn.functional.cross_entropy(
                virtual_logits, y_train, reduction="none"
            )
            virtual_weights = vnet(
                virtual_losses.unsqueeze(1)
            ).squeeze(1)
            normalized_virtual_weights = virtual_weights / (
                virtual_weights.sum() + MW_EPS
            )
            virtual_objective = torch.sum(
                normalized_virtual_weights * virtual_losses
            )

            grads = torch.autograd.grad(
                virtual_objective,
                tuple(params.values()),
                create_graph=True,
                allow_unused=False,
            )
            updated_params = OrderedDict(
                (name, parameter - MW_INNER_LR * gradient)
                for (name, parameter), gradient in zip(params.items(), grads)
            )

            state = OrderedDict()
            state.update(updated_params)
            state.update(buffers)
            meta_logits = functional_call(model, state, (x_meta,))
            meta_loss = nn.functional.cross_entropy(meta_logits, y_meta)

            vnet_optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
            vnet_optimizer.step()

        # B. Update the live classifier. This is the only forward that updates
        # its BatchNorm running statistics.
        model_optimizer.zero_grad(set_to_none=True)
        train_logits = model(x_train)
        train_losses = nn.functional.cross_entropy(
            train_logits, y_train, reduction="none"
        )

        with torch.no_grad():
            raw_weights = vnet(train_losses.unsqueeze(1)).squeeze(1)
            sample_weights = raw_weights / (raw_weights.sum() + MW_EPS)

        weighted_loss = torch.sum(sample_weights * train_losses)
        weighted_loss.backward()
        model_optimizer.step()

        with torch.no_grad():
            n = y_train.numel()
            total_weighted_loss += weighted_loss.item() * n
            total_unweighted_ce += train_losses.sum().item()
            total_correct += (train_logits.argmax(1) == y_train).sum().item()
            total_seen += n
            total_weight += raw_weights.sum().item()

        if PRINT_EVERY_BATCHES and batch_idx % PRINT_EVERY_BATCHES == 0:
            print(
                f"[Meta-Weight-Net] epoch={epoch} "
                f"batch={batch_idx}/{len(train_loader)} "
                f"meta_loss={meta_loss.item():.4f} "
                f"mean_raw_weight={raw_weights.mean().item():.4f}",
                flush=True,
            )

    return (
        total_weighted_loss / total_seen,
        total_unweighted_ce / total_seen,
        total_correct / total_seen,
        total_weight / total_seen,
        float(total_seen),
    )


def make_mwnet_term_model(in_channels, num_classes):
    model = SimpleConvNet(in_channels=in_channels, num_classes=num_classes)
    for index, layer in enumerate(model.features):
        if isinstance(layer, nn.MaxPool2d):
            model.features[index] = DifferentiableMaxPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                ceil_mode=layer.ceil_mode,
            )
    return model


# =========================================================
# One run: one seed × one noise level × one noise type
# =========================================================
def run_single(seed, noise_rate, noise_type):
    print("\n" + "=" * 90)
    print(f"Dataset = {DATASET_NAME} | Seed = {seed} | Noise type = {noise_type} | Noise rate = {noise_rate}")
    print("=" * 90, flush=True)

    set_seed(seed)

    print("Loading datasets...", flush=True)
    train_dataset = NoisyImageDataset(
        dataset_name=DATASET_NAME,
        root=ROOT_DIR,
        train=True,
        noise_rate=noise_rate,
        noise_type=noise_type,
        download=True
    )
    test_dataset = NoisyImageDataset(
        dataset_name=DATASET_NAME,
        root=ROOT_DIR,
        train=False,
        noise_rate=0.0,
        noise_type="symmetric",
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    summary_rows = []
    epoch_rows = []
    
    
    # -----------------------------------------------------
    # Darwin
    # -----------------------------------------------------
    print("\nStarting Darwin training...", flush=True)
    darwin_model = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)
    darwin_opt = optim.Adam(darwin_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    darwin_trainer = FitnessTrainer(
        model=darwin_model,
        n_samples=len(train_dataset),
        alpha=DARWIN_ALPHA,
        beta=DARWIN_BETA,
        gamma=DARWIN_GAMMA,
        ema_decay=DARWIN_EMA_DECAY,
        min_weight=DARWIN_MIN_WEIGHT,
        warmup_epochs=DARWIN_WARMUP_EPOCHS,
        weight_sharpness=DARWIN_WEIGHT_SHARPNESS,
        normalize_mean=DARWIN_NORMALIZE_MEAN,
        device=device
    )

    darwin_best = 0.0
    darwin_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = darwin_trainer.train_epoch(
            train_loader, darwin_opt, epoch=epoch, method_name="Darwin"
        )
        test_loss, test_acc = evaluate(darwin_model, test_loader, device)

        darwin_best = max(darwin_best, test_acc)
        darwin_final = test_acc

        stats = survival_stats(
            darwin_trainer.sample_weights,
            train_dataset.is_noisy,
            device=device,
            exclusion_threshold=EXCLUSION_THRESHOLD
        )

        print_epoch_result(
            "Darwin", epoch, EPOCHS,
            train_loss, train_acc, test_loss, test_acc, time.time() - t0,
            extra_text=(
                f"mean weight={stats['mean_weight_all']:.4f} | "
                f"effective fraction≈{stats['effective_fraction']:.4f} | "
                f"approx excluded≈{stats['approx_excluded_fraction']:.4f}"
            )
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "Darwin", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            **stats
        })

    darwin_final_stats = survival_stats(
        darwin_trainer.sample_weights,
        train_dataset.is_noisy,
        device=device,
        exclusion_threshold=EXCLUSION_THRESHOLD
    )

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "Darwin", "best_test_acc": darwin_best, "final_test_acc": darwin_final,
        "mean_weight_all": darwin_final_stats["mean_weight_all"],
        "clean_survival": darwin_final_stats["clean_survival"],
        "noisy_survival": darwin_final_stats["noisy_survival"],
        "effective_fraction": darwin_final_stats["effective_fraction"],
        "approx_excluded_fraction": darwin_final_stats["approx_excluded_fraction"]
    })


    # -----------------------------------------------------
    # Co-teaching
    # -----------------------------------------------------
    print("\nStarting Co-teaching training...", flush=True)
    co_model1 = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)
    co_model2 = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)

    co_opt1 = optim.Adam(co_model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    co_opt2 = optim.Adam(co_model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    forget_schedule = make_forget_rate_schedule(
        noise_rate=noise_rate,
        epochs=EPOCHS,
        num_gradual=COTEACH_NUM_GRADUAL,
        exponent=COTEACH_EXPONENT
    )

    co_best = 0.0
    co_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        forget_rate = float(forget_schedule[epoch - 1])

        train_loss, train_acc = train_coteaching_epoch(
            co_model1, co_model2, train_loader, co_opt1, co_opt2,
            device=device, forget_rate=forget_rate, epoch=epoch, method_name="Co-teaching"
        )
        test_loss, test_acc = evaluate_coteaching(co_model1, co_model2, test_loader, device)

        co_best = max(co_best, test_acc)
        co_final = test_acc

        print_epoch_result(
            "Co-teaching", epoch, EPOCHS,
            train_loss, train_acc, test_loss, test_acc, time.time() - t0
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "Co-teaching", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "mean_weight_all": np.nan, "median_weight_all": np.nan,
            "min_weight_all": np.nan, "max_weight_all": np.nan,
            "clean_survival": np.nan, "noisy_survival": np.nan,
            "clean_below_threshold": np.nan, "noisy_below_threshold": np.nan,
            "all_below_threshold": np.nan, "effective_samples": np.nan,
            "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
        })

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "Co-teaching", "best_test_acc": co_best, "final_test_acc": co_final,
        "mean_weight_all": np.nan, "clean_survival": np.nan, "noisy_survival": np.nan,
        "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
    })
    
    
    # -----------------------------------------------------
    # GCE
    # -----------------------------------------------------
    print("\nStarting GCE training...", flush=True)
    gce_model = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)
    gce_opt = optim.Adam(gce_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    gce_best = 0.0
    gce_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_gce_epoch(
            gce_model, train_loader, gce_opt, device,
            q=GCE_Q, epoch=epoch, method_name="GCE"
        )
        test_loss, test_acc = evaluate(gce_model, test_loader, device)

        gce_best = max(gce_best, test_acc)
        gce_final = test_acc

        print_epoch_result(
            "GCE", epoch, EPOCHS,
            train_loss, train_acc, test_loss, test_acc, time.time() - t0
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "GCE", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "mean_weight_all": np.nan, "median_weight_all": np.nan,
            "min_weight_all": np.nan, "max_weight_all": np.nan,
            "clean_survival": np.nan, "noisy_survival": np.nan,
            "clean_below_threshold": np.nan, "noisy_below_threshold": np.nan,
            "all_below_threshold": np.nan, "effective_samples": np.nan,
            "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
        })

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "GCE", "best_test_acc": gce_best, "final_test_acc": gce_final,
        "mean_weight_all": np.nan, "clean_survival": np.nan, "noisy_survival": np.nan,
        "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
    })
    
    
    # -----------------------------------------------------
    # Standard
    # -----------------------------------------------------
    print("\nStarting Standard training...", flush=True)
    standard_model = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)
    standard_opt = optim.Adam(standard_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    standard_best = 0.0
    standard_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_standard_epoch(
            standard_model, train_loader, standard_opt, device,
            epoch=epoch, method_name="Standard"
        )
        test_loss, test_acc = evaluate(standard_model, test_loader, device)

        standard_best = max(standard_best, test_acc)
        standard_final = test_acc

        print_epoch_result(
            "Standard", epoch, EPOCHS,
            train_loss, train_acc, test_loss, test_acc, time.time() - t0
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "Standard", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "mean_weight_all": np.nan, "median_weight_all": np.nan,
            "min_weight_all": np.nan, "max_weight_all": np.nan,
            "clean_survival": np.nan, "noisy_survival": np.nan,
            "clean_below_threshold": np.nan, "noisy_below_threshold": np.nan,
            "all_below_threshold": np.nan, "effective_samples": np.nan,
            "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
        })

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "Standard", "best_test_acc": standard_best, "final_test_acc": standard_final,
        "mean_weight_all": np.nan, "clean_survival": np.nan, "noisy_survival": np.nan,
        "effective_fraction": np.nan, "approx_excluded_fraction": np.nan
    })



    # -----------------------------------------------------
    # Random control
    # -----------------------------------------------------
    print("\nStarting Random control training...", flush=True)
    random_model = SimpleConvNet(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"]
    ).to(device)
    random_opt = optim.Adam(random_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    random_trainer = RandomWeightTrainer(
        model=random_model,
        n_samples=len(train_dataset),
        min_weight=DARWIN_MIN_WEIGHT,
        device=device
    )

    random_best = 0.0
    random_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        target_mean = darwin_trainer.sample_weights.mean().item()

        train_loss, train_acc = random_trainer.train_epoch(
            train_loader, random_opt, target_mean_weight=target_mean,
            epoch=epoch, method_name="Random control"
        )
        test_loss, test_acc = evaluate(random_model, test_loader, device)

        random_best = max(random_best, test_acc)
        random_final = test_acc

        stats = survival_stats(
            random_trainer.sample_weights,
            train_dataset.is_noisy,
            device=device,
            exclusion_threshold=EXCLUSION_THRESHOLD
        )

        print_epoch_result(
            "Random control", epoch, EPOCHS,
            train_loss, train_acc, test_loss, test_acc, time.time() - t0,
            extra_text=(
                f"mean weight={stats['mean_weight_all']:.4f} | "
                f"effective fraction≈{stats['effective_fraction']:.4f} | "
                f"approx excluded≈{stats['approx_excluded_fraction']:.4f}"
            )
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "Random control", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            **stats
        })

    random_final_stats = survival_stats(
        random_trainer.sample_weights,
        train_dataset.is_noisy,
        device=device,
        exclusion_threshold=EXCLUSION_THRESHOLD
    )

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "Random control", "best_test_acc": random_best, "final_test_acc": random_final,
        "mean_weight_all": random_final_stats["mean_weight_all"],
        "clean_survival": random_final_stats["clean_survival"],
        "noisy_survival": random_final_stats["noisy_survival"],
        "effective_fraction": random_final_stats["effective_fraction"],
        "approx_excluded_fraction": random_final_stats["approx_excluded_fraction"]
    })

    # -----------------------------------------------------
    # TERM
    # -----------------------------------------------------
    print("\nStarting TERM training...", flush=True)
    term_data = build_data(DATASET_NAME, noise_rate, seed, needs_meta=False)
    set_seed(seed)
    term_model = make_mwnet_term_model(
        in_channels=CFG["in_channels"], num_classes=CFG["num_classes"]
    ).to(DEVICE)
    term_opt = optim.Adam(term_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    term_best = 0.0
    term_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_objective, train_ce, train_acc, _ = train_term_epoch(
            term_model, term_data.train_loader, term_opt, TERM_T, epoch
        )
        test_loss, test_acc = evaluate(term_model, term_data.test_loader, DEVICE)

        term_best = max(term_best, test_acc)
        term_final = test_acc

        print_epoch_result(
            "TERM", epoch, EPOCHS,
            train_objective, train_acc, test_loss, test_acc, time.time() - t0,
            extra_text=f"cross-entropy={train_ce:.4f} | tilt={TERM_T}",
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "TERM", "epoch": epoch,
            "train_loss": train_objective, "train_acc": train_acc,
            "train_objective": train_objective, "train_ce": train_ce,
            "test_loss": test_loss, "test_acc": test_acc,
            "term_t": TERM_T, "meta_size": 0,
            "mean_weight_all": np.nan, "clean_survival": np.nan,
            "noisy_survival": np.nan, "effective_fraction": np.nan,
            "approx_excluded_fraction": np.nan,
        })

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "TERM", "best_test_acc": term_best, "final_test_acc": term_final,
        "term_t": TERM_T, "meta_size": 0,
        "mean_weight_all": np.nan, "clean_survival": np.nan,
        "noisy_survival": np.nan, "effective_fraction": np.nan,
        "approx_excluded_fraction": np.nan,
    })

    del term_model, term_opt, term_data

    # -----------------------------------------------------
    # Meta-Weight-Net
    # -----------------------------------------------------
    print("\nStarting Meta-Weight-Net training...", flush=True)
    mw_data = build_data(DATASET_NAME, noise_rate, seed, needs_meta=True)
    if mw_data.meta_loader is None:
        raise ValueError("Meta-Weight-Net requires a clean meta loader.")

    set_seed(seed)
    mw_model = make_mwnet_term_model(
        in_channels=CFG["in_channels"], num_classes=CFG["num_classes"]
    ).to(DEVICE)
    mw_vnet = MetaWeightNet(MW_HIDDEN_SIZE).to(DEVICE)
    mw_opt = optim.Adam(mw_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mw_vnet_opt = optim.Adam(mw_vnet.parameters(), lr=MW_META_LR)

    mw_best = 0.0
    mw_final = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_weighted_loss, train_ce, train_acc, mean_weight, _ = train_mwnet_epoch(
            model=mw_model,
            vnet=mw_vnet,
            train_loader=mw_data.train_loader,
            meta_loader=mw_data.meta_loader,
            model_optimizer=mw_opt,
            vnet_optimizer=mw_vnet_opt,
            epoch=epoch,
        )
        test_loss, test_acc = evaluate(mw_model, mw_data.test_loader, DEVICE)

        mw_best = max(mw_best, test_acc)
        mw_final = test_acc

        print_epoch_result(
            "Meta-Weight-Net", epoch, EPOCHS,
            train_weighted_loss, train_acc, test_loss, test_acc, time.time() - t0,
            extra_text=f"cross-entropy={train_ce:.4f} | mean raw weight={mean_weight:.4f}",
        )

        epoch_rows.append({
            "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
            "method": "MW-Net", "epoch": epoch,
            "train_loss": train_weighted_loss, "train_acc": train_acc,
            "train_objective": train_weighted_loss, "train_ce": train_ce,
            "test_loss": test_loss, "test_acc": test_acc,
            "term_t": np.nan, "meta_size": META_SIZE,
            "mean_sample_weight": mean_weight,
            "mean_weight_all": np.nan, "clean_survival": np.nan,
            "noisy_survival": np.nan, "effective_fraction": np.nan,
            "approx_excluded_fraction": np.nan,
        })

    summary_rows.append({
        "dataset": DATASET_NAME, "seed": seed, "noise_type": noise_type, "noise_rate": noise_rate,
        "method": "MW-Net", "best_test_acc": mw_best, "final_test_acc": mw_final,
        "term_t": np.nan, "meta_size": META_SIZE,
        "mean_weight_all": np.nan, "clean_survival": np.nan,
        "noisy_survival": np.nan, "effective_fraction": np.nan,
        "approx_excluded_fraction": np.nan,
    })

    del mw_model, mw_vnet, mw_opt, mw_vnet_opt, mw_data

    return summary_rows, epoch_rows


# =========================================================
# Summary printing
# =========================================================
def print_summary(df):
    print("\n" + "=" * 175)
    print(f"SUMMARY ACROSS SEEDS | DATASET = {DATASET_NAME}")
    print("=" * 175)

    methods_in_order = ["Darwin", "Co-teaching", "GCE", "MW-Net", "TERM", "Standard", "Random control"]

    for noise_type in sorted(df["noise_type"].unique()):
        print(f"\nNOISE TYPE = {noise_type}")
        print("-" * 175)

        header = (
            f"{'Noise':<8}"
            f"{'Method':<18}"
            f"{'Best test acc':<22}"
            f"{'Final test acc':<22}"
            f"{'Clean survival':<22}"
            f"{'Noisy survival':<22}"
            f"{'Eff. fraction':<18}"
        )
        print(header)
        print("-" * 175)

        sub_type = df[df["noise_type"] == noise_type]

        for noise_rate in sorted(sub_type["noise_rate"].unique()):
            sub = sub_type[sub_type["noise_rate"] == noise_rate]

            for method in methods_in_order:
                s = sub[sub["method"] == method]

                best_mean, best_sd = mean_sd(s["best_test_acc"].tolist())
                final_mean, final_sd = mean_sd(s["final_test_acc"].tolist())

                if method in ["Darwin", "Random control"]:
                    clean_mean, clean_sd = mean_sd(s["clean_survival"].tolist())
                    noisy_mean, noisy_sd = mean_sd(s["noisy_survival"].tolist())
                    eff_mean, eff_sd = mean_sd(s["effective_fraction"].tolist())

                    clean_text = f"{clean_mean:.4f} ± {clean_sd:.4f}"
                    noisy_text = f"{noisy_mean:.4f} ± {noisy_sd:.4f}"
                    eff_text = f"{eff_mean:.4f} ± {eff_sd:.4f}"
                else:
                    clean_text = "-"
                    noisy_text = "-"
                    eff_text = "-"

                print(
                    f"{noise_rate:<8.1f}"
                    f"{method:<18}"
                    f"{best_mean:.4f} ± {best_sd:.4f}{'':<6}"
                    f"{final_mean:.4f} ± {final_sd:.4f}{'':<6}"
                    f"{clean_text:<22}"
                    f"{noisy_text:<22}"
                    f"{eff_text:<18}"
                )
            print("-" * 175)


# =========================================================
# Main
# =========================================================
def main():
    overall_start = time.time()

    print("Using device:", device, flush=True)
    print(f"DATASET_NAME = {DATASET_NAME}", flush=True)
    print(f"SEEDS = {SEEDS}", flush=True)
    print(f"NOISE_TYPES = {NOISE_TYPES}", flush=True)
    print(f"NOISE_RATES = {NOISE_RATES}", flush=True)
    print(f"EPOCHS = {EPOCHS}", flush=True)
    print(f"RESULTS_CSV = {RESULTS_CSV}", flush=True)
    print(f"EPOCH_RESULTS_CSV = {EPOCH_RESULTS_CSV}", flush=True)

    print("\nDarwin parameters:", flush=True)
    print(f"  alpha = {DARWIN_ALPHA}", flush=True)
    print(f"  beta = {DARWIN_BETA}", flush=True)
    print(f"  gamma = {DARWIN_GAMMA}", flush=True)
    print(f"  ema_decay = {DARWIN_EMA_DECAY}", flush=True)
    print(f"  min_weight = {DARWIN_MIN_WEIGHT}", flush=True)
    print(f"  warmup_epochs = {DARWIN_WARMUP_EPOCHS}", flush=True)
    print(f"  weight_sharpness = {DARWIN_WEIGHT_SHARPNESS}", flush=True)
    print(f"  exclusion_threshold = {EXCLUSION_THRESHOLD}", flush=True)

    print("\nGCE parameters:", flush=True)
    print(f"  q = {GCE_Q}", flush=True)

    print(f"\nTERM tilt = {TERM_T}", flush=True)
    print(f"MW-Net clean meta-set size = {META_SIZE}", flush=True)
    print(f"MW-Net meta learning rate = {MW_META_LR}", flush=True)
    print(f"MW-Net hidden layer size = {MW_HIDDEN_SIZE}", flush=True)

    print("\nCo-teaching parameters:", flush=True)
    print(f"  num_gradual = {COTEACH_NUM_GRADUAL}", flush=True)
    print(f"  exponent = {COTEACH_EXPONENT}", flush=True)

    total_runs = len(SEEDS) * len(NOISE_RATES) * len(NOISE_TYPES)
    current_run = 0

    all_summary_rows = []
    all_epoch_rows = []

    for noise_type in NOISE_TYPES:
        for noise_rate in NOISE_RATES:
            for seed in SEEDS:
                current_run += 1
                print(
                    f"\n##### Global progress: run {current_run}/{total_runs} "
                    f"(dataset={DATASET_NAME}, seed={seed}, noise_type={noise_type}, noise={noise_rate}) #####",
                    flush=True
                )

                summary_rows, epoch_rows = run_single(seed, noise_rate, noise_type)

                all_summary_rows.extend(summary_rows)
                all_epoch_rows.extend(epoch_rows)

                df_summary_partial = pd.DataFrame(all_summary_rows)
                df_epoch_partial = pd.DataFrame(all_epoch_rows)

                df_summary_partial.to_csv(RESULTS_CSV, index=False)
                df_epoch_partial.to_csv(EPOCH_RESULTS_CSV, index=False)

                print(f"Partial summary results saved to: {RESULTS_CSV}", flush=True)
                print(f"Partial epoch results saved to: {EPOCH_RESULTS_CSV}", flush=True)

    df_summary = pd.DataFrame(all_summary_rows)
    df_epoch = pd.DataFrame(all_epoch_rows)

    df_summary.to_csv(RESULTS_CSV, index=False)
    df_epoch.to_csv(EPOCH_RESULTS_CSV, index=False)

    print_summary(df_summary)

    print(f"\nSaved detailed summary results to: {RESULTS_CSV}", flush=True)
    print(f"Saved epoch-by-epoch results to: {EPOCH_RESULTS_CSV}", flush=True)
    print(f"Total runtime: {format_time(time.time() - overall_start)}", flush=True)



if __name__ == "__main__":
    main()
