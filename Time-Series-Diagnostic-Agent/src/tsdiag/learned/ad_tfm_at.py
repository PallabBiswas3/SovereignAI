from __future__ import annotations

"""Adaptive time-frequency memory network for transformer fault diagnosis.

This module adapts the AD-TFM-AT method from Li et al.,
"Incipient Fault Detection in Power Distribution System: A Time-Frequency
Embedded Deep Learning Based Approach" (arXiv:2302.09332) to this project's
six-channel Ua/Ub/Uc/Ia/Ib/Ic waveform representation.

PyTorch is intentionally an optional dependency. Install with::

    pip install -e ".[deep]"
"""

from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "AD-TFM-AT requires the optional deep-learning dependency. "
        "Install with `pip install -e \".[deep]\"`."
    ) from exc


@dataclass(frozen=True)
class ADTFMConfig:
    input_size: int = 6
    hidden_size: int = 32
    time_dimensions: int = 4
    frequency_dimensions: int = 4
    omega0: float = 16.0
    attention_size: int = 64
    num_classes: int = 5
    dropout: float = 0.0


class ADTFMCell(nn.Module):
    """Adaptive time-frequency memory cell.

    The cell keeps real and imaginary time-frequency memories with shape
    ``[batch, hidden, K, J]``. A joint state/time/frequency forget gate controls
    the retained memory. New information is modulated by a Morlet-like adaptive
    wavelet whose scale/frequency and translation parameters are learned from
    the current input-gate state.

    The released authors' implementation uses the adaptive ``a`` term as a
    multiplier of the Morlet carrier frequency. We follow that implementation
    because it avoids the numerical singularity produced by dividing by a
    tanh-bounded value close to zero while preserving the paper's key idea:
    input-dependent adaptive time-frequency atoms.
    """

    def __init__(self, config: ADTFMConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        c = config.input_size
        k = config.time_dimensions
        j = config.frequency_dimensions

        def gate(out_features: int) -> nn.ModuleDict:
            return nn.ModuleDict({
                "x": nn.Linear(c, out_features, bias=False),
                "h": nn.Linear(d, out_features, bias=True),
            })

        self.state_forget = gate(d)
        self.time_forget = gate(k)
        self.freq_forget = gate(j)
        self.input_gate = gate(d)
        self.modulation_gate = gate(d)
        self.output_gate = gate(d)

        self.wavelet_scale = nn.Linear(d, k * j)
        self.wavelet_shift = nn.Linear(d, k * j)

        # Per-state, per-time, per-frequency projection corresponding to the
        # paper's W_c^{k,j} and b_c^{k,j} aggregation.
        self.cell_weight = nn.Parameter(torch.empty(d, k, j))
        self.cell_bias = nn.Parameter(torch.zeros(d, k, j))
        nn.init.xavier_uniform_(self.cell_weight.unsqueeze(0))

    @staticmethod
    def _gate(module: nn.ModuleDict, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return module["x"](x) + module["h"](h)

    def initial_state(self, batch_size: int, *, device, dtype):
        shape = (
            batch_size,
            self.config.hidden_size,
            self.config.time_dimensions,
            self.config.frequency_dimensions,
        )
        real = torch.zeros(shape, device=device, dtype=dtype)
        imag = torch.zeros_like(real)
        hidden = torch.zeros(batch_size, self.config.hidden_size, device=device, dtype=dtype)
        return real, imag, hidden

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        time_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        real_prev, imag_prev, h_prev = state
        cfg = self.config

        f_state = torch.sigmoid(self._gate(self.state_forget, x_t, h_prev))
        f_time = torch.sigmoid(self._gate(self.time_forget, x_t, h_prev))
        f_freq = torch.sigmoid(self._gate(self.freq_forget, x_t, h_prev))
        joint_forget = (
            f_state[:, :, None, None]
            * f_time[:, None, :, None]
            * f_freq[:, None, None, :]
        )

        i_t = torch.sigmoid(self._gate(self.input_gate, x_t, h_prev))
        g_t = torch.tanh(self._gate(self.modulation_gate, x_t, h_prev))
        input_state = i_t * g_t

        batch = x_t.shape[0]
        k = cfg.time_dimensions
        j = cfg.frequency_dimensions
        adaptive_scale = torch.tanh(self.wavelet_scale(input_state)).reshape(batch, k, j)
        adaptive_shift = torch.tanh(self.wavelet_shift(input_state)).reshape(batch, k, j)

        # Coordinates mirror the authors' released Morlet implementation:
        # z = (t + b)/2^j - k, with learnable a and b.
        k_index = torch.arange(k, device=x_t.device, dtype=x_t.dtype).view(1, k, 1)
        j_index = torch.arange(j, device=x_t.device, dtype=x_t.dtype).view(1, 1, j)
        dilation = torch.pow(torch.tensor(2.0, device=x_t.device, dtype=x_t.dtype), j_index)
        t = time_fraction.reshape(batch, 1, 1)
        z = (t + adaptive_shift) / dilation - k_index
        envelope = torch.exp(-0.5 * z.square())
        phase = cfg.omega0 * adaptive_scale * z
        wavelet_real = torch.cos(phase) * envelope
        wavelet_imag = torch.sin(phase) * envelope

        injected = input_state[:, :, None, None]
        real = joint_forget * real_prev + injected * wavelet_real[:, None, :, :]
        imag = joint_forget * imag_prev + injected * wavelet_imag[:, None, :, :]
        amplitude = torch.sqrt(real.square() + imag.square() + 1e-8)

        projected = torch.tanh(amplitude * self.cell_weight[None] + self.cell_bias[None])
        c_tilde = projected.sum(dim=(-1, -2)) / float(np.sqrt(k * j))
        o_t = torch.sigmoid(self._gate(self.output_gate, x_t, h_prev))
        hidden = o_t * torch.tanh(c_tilde)
        return hidden, (real, imag, hidden)


class ADTFMAT(nn.Module):
    """AD-TFM recurrent encoder followed by trainable context attention."""

    def __init__(self, config: ADTFMConfig | None = None):
        super().__init__()
        self.config = config or ADTFMConfig()
        self.cell = ADTFMCell(self.config)
        d = self.config.hidden_size
        a = self.config.attention_size
        self.attention_projection = nn.Linear(d, a)
        self.context_vector = nn.Parameter(torch.empty(a))
        nn.init.normal_(self.context_vector, std=0.1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(d, self.config.num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("AD-TFM input must have shape [batch, time, channels]")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"expected {self.config.input_size} input channels, got {x.shape[-1]}"
            )
        batch, steps, _ = x.shape
        state = self.cell.initial_state(batch, device=x.device, dtype=x.dtype)
        hidden_states = []
        denom = max(steps - 1, 1)
        for t in range(steps):
            time_fraction = torch.full(
                (batch,), float(t) / float(denom), device=x.device, dtype=x.dtype
            )
            hidden, state = self.cell(x[:, t], state, time_fraction)
            hidden_states.append(hidden)

        hidden = torch.stack(hidden_states, dim=1)
        u = torch.tanh(self.attention_projection(hidden))
        scores = torch.einsum("bta,a->bt", u, self.context_vector)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        return self.dropout(pooled), attention

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        pooled, attention = self.encode(x)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attention
        return logits


DEFAULT_SGAH_LABELS = (
    "single_phase_ground_fault",
    "inter_phase_short_circuit_fault",
    "two_phase_ground_fault",
    "main_transformer_fault",
    "normal",
)


def _phase_permutation_indices(phase_order: Sequence[int]) -> np.ndarray:
    if sorted(phase_order) != [0, 1, 2]:
        raise ValueError("phase_order must be a permutation of (0, 1, 2)")
    # Ua,Ub,Uc,Ia,Ib,Ic -- keep voltage/current belonging to the same phase paired.
    return np.asarray([*phase_order, *(3 + np.asarray(phase_order))], dtype=int)


def phase_switch_augment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "paper",
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-switch augmentation for six-channel three-phase waveforms.

    ``mode='paper'`` returns the original sample plus A<->B and A<->C swaps,
    matching the phase-switch examples described in the paper. ``mode='all'``
    uses all six phase permutations. Voltage/current pairs are always moved
    together, so the physical phase relationship is preserved.
    """

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError("phase switching expects X with shape [N, T, 6]")
    if len(x) != len(y):
        raise ValueError("X and y lengths must match")

    if mode == "paper":
        orders: Iterable[Sequence[int]] = ((0, 1, 2), (1, 0, 2), (2, 1, 0))
    elif mode == "all":
        orders = permutations((0, 1, 2))
    else:
        raise ValueError("mode must be 'paper' or 'all'")

    augmented = [x[:, :, _phase_permutation_indices(order)] for order in orders]
    return np.concatenate(augmented, axis=0), np.tile(y, len(augmented))


class ADTFMClassifier:
    """Callable adapter compatible with the diagnostic pipeline's model hook."""

    input_mode = "raw_waveform"

    def __init__(
        self,
        model: ADTFMAT,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        *,
        class_names: Sequence[str] = DEFAULT_SGAH_LABELS,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32).reshape(1, 1, -1)
        self.channel_std = np.asarray(channel_std, dtype=np.float32).reshape(1, 1, -1)
        self.class_names = tuple(class_names)
        if len(self.class_names) != self.model.config.num_classes:
            raise ValueError("class_names length must match num_classes")

    def _prepare(self, waveform: np.ndarray) -> torch.Tensor:
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-1] != self.model.config.input_size:
            raise ValueError("waveform must have shape [T,C] or [N,T,C]")
        x = (x - self.channel_mean) / self.channel_std
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_batch(self, waveform: np.ndarray) -> dict:
        tensor = self._prepare(waveform)
        logits, attention = self.model(tensor, return_attention=True)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        return {
            "probabilities": probabilities,
            "predicted_indices": predicted,
            "attention": attention.cpu().numpy(),
        }

    def __call__(self, waveform: np.ndarray) -> dict:
        result = self.predict_batch(waveform)
        probs = result["probabilities"][0]
        index = int(result["predicted_indices"][0])
        predicted_class = self.class_names[index]
        probability_map = {name: float(p) for name, p in zip(self.class_names, probs)}
        # The transformer's public diagnostic contract diagnoses an internal main
        # transformer fault. Other SGAH grid-fault classes stay visible in the
        # probability map but must not masquerade as a transformer diagnosis.
        label = "main_transformer_fault" if predicted_class == "main_transformer_fault" else None
        return {
            "label": label,
            "predicted_class": predicted_class,
            "confidence": float(probs[index]),
            "probabilities": probability_map,
            "attention": result["attention"][0].tolist(),
            "model_family": "AD-TFM-AT",
        }


def save_ad_tfm_classifier(classifier: ADTFMClassifier, path: str | Path) -> Path:
    """Persist a trained classifier together with its inference contract."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tsdiag.ad-tfm-at",
            "format_version": 1,
            "config": asdict(classifier.model.config),
            "state_dict": classifier.model.state_dict(),
            "channel_mean": torch.as_tensor(
                classifier.channel_mean.reshape(-1), dtype=torch.float32
            ),
            "channel_std": torch.as_tensor(
                classifier.channel_std.reshape(-1), dtype=torch.float32
            ),
            "class_names": list(classifier.class_names),
        },
        target,
    )
    return target


def load_ad_tfm_classifier(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ADTFMClassifier:
    """Load a trusted, versioned AD-TFM-AT checkpoint for inference."""

    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "tsdiag.ad-tfm-at":
        raise ValueError(f"Unsupported AD-TFM-AT checkpoint format: {source}")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported AD-TFM-AT checkpoint version: {payload.get('format_version')}")
    config = ADTFMConfig(**dict(payload["config"]))
    model = ADTFMAT(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return ADTFMClassifier(
        model,
        np.asarray(payload["channel_mean"], dtype=np.float32),
        np.asarray(payload["channel_std"], dtype=np.float32),
        class_names=tuple(payload["class_names"]),
        device=device,
    )


def train_ad_tfm_at(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    config: ADTFMConfig | None = None,
    class_names: Sequence[str] = DEFAULT_SGAH_LABELS,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    phase_augmentation: bool = True,
    augmentation_mode: str = "paper",
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[ADTFMClassifier, list[dict]]:
    """Train AD-TFM-AT with train-only normalization and optional phase switching."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = config or ADTFMConfig(num_classes=len(class_names))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    if x.ndim != 3 or x.shape[-1] != config.input_size:
        raise ValueError("train_x must have shape [N,T,input_size]")
    if len(x) != len(y):
        raise ValueError("train_x and train_y lengths must match")
    if phase_augmentation:
        x, y = phase_switch_augment(x, y, mode=augmentation_mode)

    channel_mean = x.mean(axis=(0, 1), keepdims=True)
    channel_std = x.std(axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std > 1e-6, channel_std, 1.0)
    x_norm = (x - channel_mean) / channel_std

    dataset = TensorDataset(
        torch.as_tensor(x_norm, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = ADTFMAT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    def evaluate(eval_x: np.ndarray, eval_y: np.ndarray) -> tuple[float, float]:
        model.eval()
        arr = np.asarray(eval_x, dtype=np.float32)
        labels = np.asarray(eval_y, dtype=np.int64)
        arr = (arr - channel_mean) / channel_std
        with torch.no_grad():
            inputs = torch.as_tensor(arr, dtype=torch.float32, device=device)
            targets = torch.as_tensor(labels, dtype=torch.long, device=device)
            logits = model(inputs)
            loss = float(criterion(logits, targets).item())
            accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
        return loss, accuracy

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_n = int(targets.numel())
            total_loss += float(loss.item()) * batch_n
            total += batch_n
            correct += int((logits.argmax(dim=1) == targets).sum().item())

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
        }
        if val_x is not None and val_y is not None and len(val_x):
            val_loss, val_accuracy = evaluate(val_x, val_y)
            row.update({"val_loss": val_loss, "val_accuracy": val_accuracy})
        history.append(row)

    wrapper = ADTFMClassifier(
        model,
        channel_mean.reshape(-1),
        channel_std.reshape(-1),
        class_names=class_names,
        device=device,
    )
    return wrapper, history
