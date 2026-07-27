from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flashvad.checkpoint import load_checkpoint_data


def test_checkpoint_loader_uses_safe_weights_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {"config": {}, "model": {}}

    monkeypatch.setattr(torch, "load", fake_load)

    checkpoint = load_checkpoint_data(tmp_path / "model.pt")

    assert checkpoint == {"config": {}, "model": {}}
    assert calls == [
        {
            "path": tmp_path / "model.pt",
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "mapping"),
        ({"model": {}}, "config"),
        ({"config": {}, "model": []}, "model"),
    ],
)
def test_checkpoint_loader_rejects_invalid_payloads(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(ValueError, match=message):
        load_checkpoint_data("invalid.pt")
