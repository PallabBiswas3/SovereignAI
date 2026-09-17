from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis


PADERBORN_6203_FREQUENCY_MULTIPLIERS = {
    "BPFO": 3.05,
    "BPFI": 4.93,
    "BSF": 1.99,
    "FTF": 0.381,
}


def bearing_fault_frequencies(
    shaft_rate_hz: float,
    multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return characteristic bearing frequencies from shaft speed.

    The defaults are the documented approximate order values for the FAG
    6203 bearing used by Paderborn. Callers should supply exact geometry-derived
    multipliers when they are available for another rig.
    """
    shaft = float(shaft_rate_hz)
    if not np.isfinite(shaft) or shaft <= 0:
        raise ValueError("shaft_rate_hz must be positive and finite")
    return {
        str(name): shaft * float(multiplier)
        for name, multiplier in (multipliers or PADERBORN_6203_FREQUENCY_MULTIPLIERS).items()
    }


def _spectral_amplitude(signal: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float).ravel()
    x = np.nan_to_num(x - np.nanmean(x))
    scale = float(np.std(x))
    if scale > 1e-12:
        x = x / scale
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) / max(x.size / 2.0, 1.0)
    return np.fft.rfftfreq(x.size, 1.0 / float(sampling_rate_hz)), spectrum


def _band_peak(freq: np.ndarray, amplitude: np.ndarray, target: float, width: float) -> float:
    mask = np.abs(freq - float(target)) <= float(width)
    if not np.any(mask):
        return 0.0
    floor = max(float(np.median(amplitude[1:])), 1e-12)
    return float(np.log1p(np.max(amplitude[mask]) / floor))


def speed_assisted_physics_features(
    channels,
    sampling_rate_hz: float,
    shaft_rate_hz: float,
    *,
    fault_frequencies: dict[str, float] | None = None,
    electrical_frequency_hz: float = 50.0,
    envelope_channel: int | None = 0,
) -> np.ndarray:
    """Extract time, shaft-order, bearing-frequency and current-sideband features.

    Exactly two signal channels are returned to the learner. A missing second
    channel is represented by zeros, which keeps Paderborn and Lenze feature
    contracts identical without pretending that their modalities are identical.
    """
    x = np.asarray(channels, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] < 32:
        raise ValueError("channels must be [channels, samples] with at least 32 samples")
    if x.shape[0] > 2:
        x = x[:2]
    if x.shape[0] == 1:
        x = np.vstack([x, np.zeros_like(x)])
    fs = float(sampling_rate_hz)
    shaft = float(shaft_rate_hz)
    faults = dict(fault_frequencies or bearing_fault_frequencies(shaft))
    width = max(fs / x.shape[1] * 1.5, shaft * 0.03, 0.5)

    values: list[float] = []
    spectra: list[tuple[np.ndarray, np.ndarray]] = []
    for signal in x:
        centered = np.nan_to_num(signal - np.nanmean(signal))
        std = max(float(np.std(centered)), 1e-12)
        normalized = centered / std
        rms = float(np.sqrt(np.mean(centered**2)))
        peak = float(np.max(np.abs(centered)))
        freq, amplitude = _spectral_amplitude(normalized, fs)
        spectra.append((freq, amplitude))
        p = amplitude[1:] ** 2
        p = p / max(float(np.sum(p)), 1e-12)
        entropy = -float(np.sum(p * np.log(p + 1e-12))) / max(np.log(max(p.size, 2)), 1.0)
        values.extend([
            float(np.log1p(rms)),
            float(kurtosis(normalized, fisher=False, bias=False)),
            float(peak / max(rms, 1e-12)),
            entropy,
        ])
        values.extend(_band_peak(freq, amplitude, order * shaft, width) for order in range(1, 9))
        for base in faults.values():
            values.extend(_band_peak(freq, amplitude, h * base, width) for h in range(1, 4))

    if envelope_channel is None or not 0 <= int(envelope_channel) < x.shape[0]:
        values.extend([0.0] * 12)
    else:
        envelope = np.abs(hilbert(np.nan_to_num(x[int(envelope_channel)] - np.nanmean(x[int(envelope_channel)]))))
        ef, ea = _spectral_amplitude(envelope, fs)
        for base in faults.values():
            values.extend(_band_peak(ef, ea, h * base, width) for h in range(1, 4))

    current_freq, current_amp = spectra[1]
    for base in faults.values():
        values.extend(
            _band_peak(current_freq, current_amp, abs(float(electrical_frequency_hz) + sign * base), width)
            for sign in (-1, 1)
        )
    values.append(float(np.log1p(shaft)))
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def normalize_waveform_channels(channels, output_samples: int = 4096) -> np.ndarray:
    x = np.asarray(channels, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("channels must be [channels, samples]")
    if x.shape[0] == 1:
        x = np.vstack([x, np.zeros_like(x)])
    x = x[:2]
    target = int(output_samples)
    if target < 128:
        raise ValueError("output_samples must be at least 128")
    positions = np.linspace(0, x.shape[1] - 1, target)
    old = np.arange(x.shape[1])
    out = np.vstack([np.interp(positions, old, np.nan_to_num(row)) for row in x])
    out -= out.mean(axis=1, keepdims=True)
    out /= np.maximum(out.std(axis=1, keepdims=True), 1e-6)
    return out.astype(np.float32)


@dataclass(frozen=True)
class BearingHybridConfig:
    mode: str = "fusion"
    input_channels: int = 2
    waveform_samples: int = 4096
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 17


class BearingHybridClassifier:
    """Two-block 1D CNN fused with explicit speed-assisted physics features."""

    FORMAT = "tsdiag-bearing-cnn-physics"
    VERSION = 1

    def __init__(self, config: BearingHybridConfig | None = None):
        self.config = config or BearingHybridConfig()
        if self.config.mode not in {"physics", "cnn", "fusion"}:
            raise ValueError("mode must be 'physics', 'cnn', or 'fusion'")
        self.classes_: np.ndarray | None = None
        self.physics_mean_: np.ndarray | None = None
        self.physics_scale_: np.ndarray | None = None
        self.model_ = None

    @staticmethod
    def _torch():
        try:
            import torch
            return torch
        except ImportError as exc:  # pragma: no cover - exercised without deep extra
            raise RuntimeError("PyTorch is required; install the project with the 'deep' extra") from exc

    def _build(self, physics_features: int, classes: int):
        torch = self._torch()
        nn = torch.nn
        mode = self.config.mode

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(2, 16, kernel_size=9, stride=2, padding=4), nn.ReLU(), nn.AvgPool1d(4),
                    nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.AvgPool1d(4),
                    nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
                )
                self.physics = nn.Sequential(nn.Linear(physics_features, 32), nn.ReLU())
                width = 32 if mode == "physics" else 64 if mode == "cnn" else 96
                self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(width, classes))

            def forward(self, waveform, physics):
                if mode == "physics":
                    features = self.physics(physics)
                elif mode == "cnn":
                    features = self.cnn(waveform)
                else:
                    features = torch.cat([self.cnn(waveform), self.physics(physics)], dim=1)
                return self.classifier(features)

        return Network()

    def fit(
        self,
        waveforms,
        physics_features,
        labels: Iterable[str],
        *,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> "BearingHybridClassifier":
        torch = self._torch()
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        w = np.asarray(waveforms, dtype=np.float32)
        p = np.asarray(physics_features, dtype=np.float32)
        y_text = np.asarray(list(labels), dtype=object)
        if w.ndim != 3 or w.shape[1] != 2 or p.ndim != 2 or len(w) != len(p) or len(w) != len(y_text):
            raise ValueError("expected waveforms [N,2,L], physics [N,P], and N labels")
        self.classes_, y = np.unique(y_text, return_inverse=True)
        self.physics_mean_ = p.mean(axis=0)
        self.physics_scale_ = np.maximum(p.std(axis=0), 1e-6)
        p = (p - self.physics_mean_) / self.physics_scale_
        self.model_ = self._build(p.shape[1], len(self.classes_))
        counts = np.bincount(y, minlength=len(self.classes_))
        weights = len(y) / np.maximum(counts * len(self.classes_), 1)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.config.learning_rate)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(w), torch.from_numpy(p.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, generator=generator)
        self.model_.train()
        epochs = max(1, int(self.config.epochs))
        training_started = perf_counter()
        for epoch in range(epochs):
            epoch_started = perf_counter()
            loss_sum = 0.0
            sample_count = 0
            for batch_w, batch_p, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(batch_w, batch_p), batch_y)
                loss.backward()
                optimizer.step()
                batch_count = int(batch_y.shape[0])
                loss_sum += float(loss.detach()) * batch_count
                sample_count += batch_count
            if progress_callback is not None:
                elapsed = perf_counter() - training_started
                epoch_seconds = perf_counter() - epoch_started
                completed = epoch + 1
                progress_callback({
                    "epoch": completed,
                    "epochs": epochs,
                    "loss": loss_sum / max(sample_count, 1),
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / completed * (epochs - completed),
                })
        self.model_.eval()
        return self

    def predict_proba(self, waveforms, physics_features) -> np.ndarray:
        if self.model_ is None or self.physics_mean_ is None or self.physics_scale_ is None:
            raise RuntimeError("classifier is not fitted")
        torch = self._torch()
        w = np.asarray(waveforms, dtype=np.float32)
        p = (np.asarray(physics_features, dtype=np.float32) - self.physics_mean_) / self.physics_scale_
        with torch.no_grad():
            return torch.softmax(self.model_(torch.from_numpy(w), torch.from_numpy(p.astype(np.float32))), dim=1).numpy()

    def predict(self, waveforms, physics_features) -> np.ndarray:
        probabilities = self.predict_proba(waveforms, physics_features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def save(self, path: str | Path) -> Path:
        if self.model_ is None:
            raise RuntimeError("classifier is not fitted")
        torch = self._torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": self.FORMAT,
            "version": self.VERSION,
            "config": asdict(self.config),
            "classes": self.classes_.tolist(),
            "physics_mean": self.physics_mean_,
            "physics_scale": self.physics_scale_,
            "state_dict": self.model_.state_dict(),
        }, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BearingHybridClassifier":
        torch = cls._torch()
        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format") != cls.FORMAT or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported bearing model artifact")
        instance = cls(BearingHybridConfig(**payload["config"]))
        instance.classes_ = np.asarray(payload["classes"], dtype=object)
        instance.physics_mean_ = np.asarray(payload["physics_mean"], dtype=np.float32)
        instance.physics_scale_ = np.asarray(payload["physics_scale"], dtype=np.float32)
        instance.model_ = instance._build(instance.physics_mean_.size, instance.classes_.size)
        instance.model_.load_state_dict(payload["state_dict"])
        instance.model_.eval()
        return instance
