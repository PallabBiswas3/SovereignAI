from __future__ import annotations

"""Controlled TFM / AD-TFM / TFM-AT / AD-TFM-AT ablation variants.

The paper defines the ablation structure but does not state the numeric fixed
scale/translation constants used by TFM in the text. For the fixed-wavelet
variants here we therefore use a controlled Morlet bank with ``a=1`` and
``b=0`` while preserving the paper's K/J grid, dilation ``2**j`` and carrier
``omega0``. This choice is explicit in benchmark output and documentation; it
is not claimed as an unrevealed paper hyperparameter.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ad_tfm_at import (
    ADTFMCell,
    ADTFMClassifier,
    ADTFMConfig,
    DEFAULT_SGAH_LABELS,
    phase_switch_augment,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "AD-TFM ablations require the optional deep-learning dependency. "
        "Install with `pip install -e \".[deep]\"`."
    ) from exc


@dataclass(frozen=True)
class TFMVariantSpec:
    name: str
    adaptive_wavelet: bool
    attention: bool


PAPER_ABLATION_VARIANTS: tuple[TFMVariantSpec, ...] = (
    TFMVariantSpec("TFM", adaptive_wavelet=False, attention=False),
    TFMVariantSpec("AD-TFM", adaptive_wavelet=True, attention=False),
    TFMVariantSpec("TFM-AT", adaptive_wavelet=False, attention=True),
    TFMVariantSpec("AD-TFM-AT", adaptive_wavelet=True, attention=True),
)


def get_variant_spec(name: str) -> TFMVariantSpec:
    normalized = name.strip().upper()
    for spec in PAPER_ABLATION_VARIANTS:
        if spec.name.upper() == normalized:
            return spec
    raise KeyError(f"unknown TFM ablation variant {name!r}")


class AblationTFMCell(ADTFMCell):
    """AD-TFM cell that can switch adaptive wavelet parameters off."""

    def __init__(
        self,
        config: ADTFMConfig,
        *,
        adaptive_wavelet: bool,
        fixed_scale: float = 1.0,
        fixed_shift: float = 0.0,
    ):
        super().__init__(config)
        self.adaptive_wavelet = bool(adaptive_wavelet)
        self.fixed_scale = float(fixed_scale)
        self.fixed_shift = float(fixed_shift)
        if not self.adaptive_wavelet:
            # Keep the same implementation surface while excluding the adaptive
            # parameter generators from optimization/parameter-count comparisons.
            for parameter in self.wavelet_scale.parameters():
                parameter.requires_grad_(False)
            for parameter in self.wavelet_shift.parameters():
                parameter.requires_grad_(False)

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
        if self.adaptive_wavelet:
            scale = torch.tanh(self.wavelet_scale(input_state)).reshape(batch, k, j)
            shift = torch.tanh(self.wavelet_shift(input_state)).reshape(batch, k, j)
        else:
            scale = torch.full(
                (batch, k, j), self.fixed_scale, device=x_t.device, dtype=x_t.dtype
            )
            shift = torch.full_like(scale, self.fixed_shift)

        k_index = torch.arange(k, device=x_t.device, dtype=x_t.dtype).view(1, k, 1)
        j_index = torch.arange(j, device=x_t.device, dtype=x_t.dtype).view(1, 1, j)
        dilation = torch.pow(torch.tensor(2.0, device=x_t.device, dtype=x_t.dtype), j_index)
        t = time_fraction.reshape(batch, 1, 1)
        z = (t + shift) / dilation - k_index
        envelope = torch.exp(-0.5 * z.square())
        phase = cfg.omega0 * scale * z
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


class TFMVariantModel(nn.Module):
    """One of the four paper ablation networks under a common implementation."""

    def __init__(
        self,
        config: ADTFMConfig,
        spec: TFMVariantSpec,
        *,
        fixed_scale: float = 1.0,
        fixed_shift: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.spec = spec
        self.cell = AblationTFMCell(
            config,
            adaptive_wavelet=spec.adaptive_wavelet,
            fixed_scale=fixed_scale,
            fixed_shift=fixed_shift,
        )
        d = config.hidden_size
        self.attention_projection = nn.Linear(d, config.attention_size) if spec.attention else None
        self.context_vector = nn.Parameter(torch.empty(config.attention_size)) if spec.attention else None
        if self.context_vector is not None:
            nn.init.normal_(self.context_vector, std=0.1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(d, config.num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("TFM input must have shape [batch, time, channels]")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(f"expected {self.config.input_size} channels, got {x.shape[-1]}")

        batch, steps, _ = x.shape
        state = self.cell.initial_state(batch, device=x.device, dtype=x.dtype)
        hidden_states = []
        denom = max(steps - 1, 1)
        for step in range(steps):
            time_fraction = torch.full(
                (batch,), float(step) / float(denom), device=x.device, dtype=x.dtype
            )
            hidden, state = self.cell(x[:, step], state, time_fraction)
            hidden_states.append(hidden)
        hidden = torch.stack(hidden_states, dim=1)

        if self.spec.attention:
            u = torch.tanh(self.attention_projection(hidden))
            scores = torch.einsum("bta,a->bt", u, self.context_vector)
            attention = torch.softmax(scores, dim=1)
            pooled = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        else:
            pooled = hidden[:, -1]
            attention = torch.zeros(batch, steps, device=x.device, dtype=x.dtype)
            attention[:, -1] = 1.0
        return self.dropout(pooled), attention

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        pooled, attention = self.encode(x)
        logits = self.classifier(pooled)
        return (logits, attention) if return_attention else logits


class TFMVariantClassifier(ADTFMClassifier):
    def __init__(self, model: TFMVariantModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.variant_name = model.spec.name

    def __call__(self, waveform: np.ndarray) -> dict:
        out = super().__call__(waveform)
        out["model_family"] = self.variant_name
        return out


def train_tfm_variant(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    variant: str | TFMVariantSpec,
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
    fixed_scale: float = 1.0,
    fixed_shift: float = 0.0,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[TFMVariantClassifier, list[dict]]:
    """Train one controlled paper-ablation variant with a shared protocol."""

    spec = get_variant_spec(variant) if isinstance(variant, str) else variant
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TFMVariantModel(
        config,
        spec,
        fixed_scale=fixed_scale,
        fixed_shift=fixed_shift,
    ).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    def evaluate(eval_x: np.ndarray, eval_y: np.ndarray) -> tuple[float, float]:
        model.eval()
        arr = (np.asarray(eval_x, dtype=np.float32) - channel_mean) / channel_std
        labels = np.asarray(eval_y, dtype=np.int64)
        with torch.no_grad():
            inputs = torch.as_tensor(arr, dtype=torch.float32, device=device)
            targets = torch.as_tensor(labels, dtype=torch.long, device=device)
            logits = model(inputs)
            return (
                float(criterion(logits, targets).item()),
                float((logits.argmax(dim=1) == targets).float().mean().item()),
            )

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
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            count = int(targets.numel())
            total_loss += float(loss.item()) * count
            total += count
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

    wrapper = TFMVariantClassifier(
        model,
        channel_mean.reshape(-1),
        channel_std.reshape(-1),
        class_names=class_names,
        device=device,
    )
    return wrapper, history


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
