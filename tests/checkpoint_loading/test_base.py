"""Tests for versatil.checkpoint_loading.base module."""

import io
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from versatil.checkpoint_loading.base import (
    BaseCheckpointLoader,
    versatil_checkpoint_safe_globals,
)
from versatil.configs import TrainingConfig


class _ArbitraryCodeExecution:
    def __reduce__(self):
        return (print, ("arbitrary code executed",))


@pytest.mark.unit
class TestVersatilCheckpointSafeGlobals:
    def test_checkpoint_with_config_hyperparameters_loads(self):
        checkpoint = {
            "state_dict": {"layer.weight": torch.ones(2, 2)},
            "hyper_parameters": OmegaConf.structured(TrainingConfig()),
            "epoch": 3,
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        with torch.serialization.safe_globals(versatil_checkpoint_safe_globals()):
            loaded = torch.load(buffer, weights_only=True)

        assert sorted(loaded.keys()) == ["epoch", "hyper_parameters", "state_dict"]
        torch.testing.assert_close(
            loaded["state_dict"]["layer.weight"],
            checkpoint["state_dict"]["layer.weight"],
        )

    def test_malicious_pickle_is_rejected(self):
        buffer = io.BytesIO()
        torch.save({"state_dict": _ArbitraryCodeExecution()}, buffer)
        buffer.seek(0)

        with (
            torch.serialization.safe_globals(versatil_checkpoint_safe_globals()),
            pytest.raises(Exception, match="Weights only load failed"),
        ):
            torch.load(buffer, weights_only=True)


@pytest.mark.unit
class TestLoadConfigDeviceOverride:
    def test_saved_device_entries_follow_the_loader_device(self, tmp_path) -> None:
        config_path = tmp_path / "config.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "experiment": {"device": "cuda"},
                    "policy": {"device": "cuda", "decoder": {"device": "cuda"}},
                    "task": {"name": "unit"},
                }
            ),
            config_path,
        )
        loader = BaseCheckpointLoader(
            device=torch.device("cpu"), checkpoint_path=str(tmp_path)
        )
        with (
            patch("versatil.checkpoint_loading.base.hydra.utils.instantiate") as inst,
            patch("versatil.checkpoint_loading.base.validate_experiment"),
        ):
            loader._load_config(config_path=str(config_path))

        instantiated = inst.call_args.args[0]
        assert instantiated.experiment.device == "cpu"
        assert instantiated.policy.device == "cpu"
        assert instantiated.policy.decoder.device == "cpu"
