from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint_data
from .config import ProjectConfig
from .evaluation import calibrate_detector
from .features import CausalFeatureExtractor
from .losses import binary_focal_loss_with_logits
from .manifest import VadDataset
from .metrics import best_f1_threshold, binary_metrics
from .model import FlashVad
from .region import RegionProfile, region_sampler


def select_training_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _boundary_loss(probabilities: Tensor, labels: Tensor) -> Tensor:
    if probabilities.shape[1] < 2:
        return probabilities.new_zeros(())
    predicted_change = torch.abs(probabilities[:, 1:] - probabilities[:, :-1])
    target_change = torch.abs(labels[:, 1:] - labels[:, :-1])
    weights = 1.0 + 4.0 * target_change
    return (torch.abs(predicted_change - target_change) * weights).mean()


@torch.inference_mode()
def _validation_sequences(
    frontend: CausalFeatureExtractor,
    model: FlashVad,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for audio, labels, _ in loader:
        features = frontend(audio.to(device))
        logits = model(features)["speech_logits"]
        usable = min(logits.shape[1], labels.shape[1])
        probabilities = torch.sigmoid(logits[:, :usable]).cpu().numpy()
        targets = (labels[:, :usable].numpy() >= 0.5).astype(np.float32)
        all_probabilities.extend(probabilities)
        all_labels.extend(targets)
    return all_probabilities, all_labels


@torch.inference_mode()
def evaluate(
    frontend: CausalFeatureExtractor,
    model: FlashVad,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], float]:
    all_probabilities, all_labels = _validation_sequences(
        frontend,
        model,
        loader,
        device,
    )
    probabilities = np.concatenate([item.reshape(-1) for item in all_probabilities])
    targets = np.concatenate([item.reshape(-1) for item in all_labels])
    threshold, metrics = best_f1_threshold(probabilities, targets)
    metrics["threshold"] = threshold
    metrics["fixed_0_5_f1"] = binary_metrics(probabilities, targets, 0.5)["f1"]
    return metrics, threshold


def train(
    config_path: str | Path,
    train_manifest: str | Path,
    valid_manifest: str | Path,
    output_dir: str | Path,
    device_name: str | None = None,
    region_profile_path: str | Path | None = None,
    noise_manifest_path: str | Path | None = None,
) -> Path:
    config = ProjectConfig.load(config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_training_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")

    frontend = CausalFeatureExtractor(config.feature).to(device)
    model = FlashVad(config.model).to(device)
    training_data = VadDataset(
        train_manifest,
        config.feature,
        config.training.chunk_seconds,
        augment=True,
        seed=config.seed,
        noise_manifest=noise_manifest_path,
    )
    valid_data = VadDataset(
        valid_manifest,
        config.feature,
        config.training.chunk_seconds,
        augment=False,
        seed=config.seed,
    )
    sampler = None
    if region_profile_path:
        sampler = region_sampler(
            training_data.items,
            RegionProfile.load(region_profile_path),
            config.seed,
        )
    train_loader = DataLoader(
        training_data,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs, 1),
    )
    positive_weight = torch.tensor(config.training.positive_weight, device=device)
    best_f1 = -1.0
    best_path = output / "best.pt"
    history: list[dict[str, float | int | str]] = []

    print(
        f"device={device} parameters={model.parameter_count:,} "
        f"train_items={len(training_data)} valid_items={len(valid_data)}"
    )
    for epoch in range(1, config.training.epochs + 1):
        started = time.perf_counter()
        training_data.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for audio, labels, auxiliary_labels in train_loader:
            audio = audio.to(device)
            labels = labels.to(device)
            auxiliary_labels = auxiliary_labels.to(device)
            features = frontend(audio)
            outputs = model(features)
            logits = outputs["speech_logits"]
            usable = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :usable]
            labels = labels[:, :usable]
            auxiliary_logits = outputs["auxiliary_logits"][:, :usable]
            auxiliary_labels = auxiliary_labels[:, :usable]
            probabilities = torch.sigmoid(logits)
            classification = binary_focal_loss_with_logits(
                logits,
                labels,
                gamma=config.training.focal_gamma,
                positive_weight=positive_weight,
            )
            boundary = _boundary_loss(probabilities, labels)
            auxiliary = F.binary_cross_entropy_with_logits(
                auxiliary_logits,
                auxiliary_labels,
            )
            loss = (
                classification
                + config.training.boundary_weight * boundary
                + config.training.auxiliary_weight * auxiliary
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics, threshold = evaluate(frontend, model, valid_loader, device)
        elapsed = time.perf_counter() - started
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "elapsed_seconds": elapsed,
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "threshold": threshold,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                best_path,
            )

    best_checkpoint = load_checkpoint_data(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    probability_sequences, label_sequences = _validation_sequences(
        frontend,
        model,
        valid_loader,
        device,
    )
    detector_config, detector_metrics = calibrate_detector(
        probability_sequences,
        label_sequences,
        config.feature.hop_ms,
        max_false_alarm_rate=config.training.detector_max_false_alarm_rate,
    )
    calibrated_config = replace(config, detector=detector_config)
    best_checkpoint["config"] = calibrated_config.to_dict()
    best_checkpoint["detector_metrics"] = detector_metrics
    torch.save(best_checkpoint, best_path)
    calibrated_config.save(output / "config.json")
    with (output / "detector-calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_split": "validation",
                "max_false_alarm_rate": config.training.detector_max_false_alarm_rate,
                "detector": asdict(detector_config),
                "metrics": detector_metrics,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    return best_path
