from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


class CausalDepthwiseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cache_size = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def _post(self, residual: Tensor, encoded: Tensor) -> Tensor:
        encoded = encoded.transpose(1, 2)
        encoded = self.norm(encoded)
        encoded = F.silu(encoded)
        return residual + self.dropout(encoded)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        channel_first = inputs.transpose(1, 2)
        encoded = self.depthwise(F.pad(channel_first, (self.cache_size, 0)))
        encoded = self.pointwise(encoded)
        return self._post(residual, encoded)

    def initial_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            batch_size, self.depthwise.in_channels, self.cache_size, device=device, dtype=dtype
        )

    def stream(self, inputs: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        # inputs: [batch, 1, channels]
        residual = inputs
        current = inputs.transpose(1, 2)
        context = torch.cat((cache, current), dim=-1)
        encoded = self.pointwise(self.depthwise(context))
        next_cache = context[..., -self.cache_size :]
        return self._post(residual, encoded), next_cache


@dataclass
class StreamingModelState:
    convolution: tuple[Tensor, ...]
    recurrent: Tensor


class FlashVad(nn.Module):
    """Small causal frame model.

    The auxiliary three-way head represents speech, music, and other vocal
    events. Binary training can ignore it; richer datasets can supervise it.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.input_norm = nn.LayerNorm(self.config.feature_dim)
        self.input_projection = nn.Linear(self.config.feature_dim, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                dilation,
                self.config.dropout,
            )
            for dilation in self.config.dilations
        )
        self.recurrent = nn.GRU(
            self.config.hidden_dim,
            self.config.recurrent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.config.recurrent_dim)
        self.speech_head = nn.Linear(self.config.recurrent_dim, 1)
        self.auxiliary_head = nn.Linear(self.config.recurrent_dim, 3)

    def _input(self, features: Tensor) -> Tensor:
        return F.silu(self.input_projection(self.input_norm(features)))

    def _heads(self, encoded: Tensor) -> dict[str, Tensor]:
        encoded = self.output_norm(encoded)
        return {
            "speech_logits": self.speech_head(encoded).squeeze(-1),
            "auxiliary_logits": self.auxiliary_head(encoded),
        }

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        encoded = self._input(features)
        for block in self.blocks:
            encoded = block(encoded)
        encoded, _ = self.recurrent(encoded)
        return self._heads(encoded)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> StreamingModelState:
        target = torch.device(device)
        caches = tuple(block.initial_cache(batch_size, target, dtype) for block in self.blocks)
        recurrent = torch.zeros(
            1, batch_size, self.config.recurrent_dim, device=target, dtype=dtype
        )
        return StreamingModelState(caches, recurrent)

    def stream_step(
        self,
        feature: Tensor,
        state: StreamingModelState,
    ) -> tuple[dict[str, Tensor], StreamingModelState]:
        if feature.ndim == 2:
            feature = feature.unsqueeze(1)
        if feature.ndim != 3 or feature.shape[1] != 1:
            raise ValueError("stream_step expects [batch, features] or [batch, 1, features]")
        encoded = self._input(feature)
        next_caches: list[Tensor] = []
        for block, cache in zip(self.blocks, state.convolution, strict=True):
            encoded, next_cache = block.stream(encoded, cache)
            next_caches.append(next_cache)
        encoded, recurrent = self.recurrent(encoded, state.recurrent)
        return self._heads(encoded), StreamingModelState(tuple(next_caches), recurrent)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
