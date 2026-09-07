"""Tests for the Tip 4 benchmark's target construction and checkpoint lookup."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from versatil.analysis.tip4_speed.benchmark import (
    CONTINUOUS_STEPS,
    find_checkpoint_dir,
    infer_method,
    target_from_cell,
    target_from_checkpoint_dir,
    unique_stage_cells,
)


class TestUniqueStageCells:
    def test_noise_and_rate_stages_dedupe_to_56_cells(self) -> None:
        cells = unique_stage_cells(stages=["final_conditional", "rate_conditional_s0"])

        # 4 methods x 4 sigma x 3 replicates, plus 4 methods x T in {120, 240}
        # (the rate stage's T=60 cells coincide with the sigma=1 seed-0 cells).
        assert len(cells) == 56
        assert len({cell.name for cell in cells}) == 56

    def test_cell_filter_restricts_by_substring(self) -> None:
        cells = unique_stage_cells(stages=["final_conditional"], cell_filter="__qfat__")

        assert len(cells) == 12
        assert all(cell.method == "qfat" for cell in cells)


class TestTargetFromCell:
    def test_continuous_arms_get_depth_and_random_init(self, tmp_path: Path) -> None:
        cells = unique_stage_cells(stages=["final_conditional"])
        qfat = next(cell for cell in cells if cell.method == "qfat")
        bcat = next(cell for cell in cells if cell.method == "bcat")
        fast = next(cell for cell in cells if cell.method == "fast")

        qfat_target = target_from_cell(cell=qfat, checkpoint_dir=tmp_path)
        bcat_target = target_from_cell(cell=bcat, checkpoint_dir=tmp_path)
        fast_target = target_from_cell(cell=fast, checkpoint_dir=tmp_path)

        assert qfat_target.continuous_steps == qfat.prediction_horizon
        assert bcat_target.continuous_steps == 1
        assert fast_target.continuous_steps is None
        assert qfat_target.allow_random_init
        assert not fast_target.allow_random_init
        assert CONTINUOUS_STEPS["bcat"](60) == 1


class TestInferMethod:
    def _config(self, decoder_target: str, discretizer_type: str | None = None):
        tokenization = (
            {}
            if discretizer_type is None
            else {
                "tokenization": {
                    "action_tokenizer": {
                        "action_discretizer": {"type": discretizer_type}
                    }
                }
            }
        )
        return OmegaConf.create(
            {
                "policy": {"decoder": {"_target_": decoder_target}},
                "task": {"dataloader": tokenization},
            }
        )

    def test_infers_all_four_arms(self) -> None:
        qfat = "versatil.models.decoding.decoders.factory.qfat.QFATActionTransformer"
        gpt = (
            "versatil.models.decoding.decoders.factory."
            "gpt_action_transformer.GPTActionTransformer"
        )
        bcat = (
            "versatil.models.decoding.decoders.factory."
            "action_transformer.SimpleActionTransformer"
        )
        assert infer_method(self._config(qfat)) == "qfat"
        assert infer_method(self._config(gpt, "fast")) == "fast"
        assert infer_method(self._config(gpt, "binned")) == "binned"
        assert infer_method(self._config(bcat)) == "bcat"

    def test_unknown_decoder_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer"):
            infer_method(self._config("some.module.DiffusionPolicy"))


class TestTargetFromCheckpointDir:
    def _write_config(
        self,
        directory: Path,
        decoder_target: str,
        horizon: int,
        discretizer_type: str | None = None,
    ) -> None:
        tokenization = (
            {}
            if discretizer_type is None
            else {
                "tokenization": {
                    "action_tokenizer": {
                        "action_discretizer": {"type": discretizer_type}
                    }
                }
            }
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "policy": {"decoder": {"_target_": decoder_target}},
                    "task": {
                        "prediction_horizon": horizon,
                        "dataloader": tokenization,
                    },
                }
            ),
            directory / "config.yaml",
        )

    def test_reads_method_and_horizon_from_config(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "tip4_matched"
        checkpoint_dir.mkdir()
        self._write_config(
            directory=checkpoint_dir,
            decoder_target="factory.qfat.QFATActionTransformer",
            horizon=1,
        )

        target = target_from_checkpoint_dir(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="last.ckpt",
            task_label="libero",
        )

        assert target.method == "qfat"
        assert target.trajectory_length == 1
        assert target.continuous_steps == 1
        assert target.task == "libero"
        assert not target.allow_random_init
        assert target.name == "tip4_matched"

    def test_tokenized_checkpoint_has_no_assigned_depth(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "gpt"
        checkpoint_dir.mkdir()
        self._write_config(
            directory=checkpoint_dir,
            decoder_target="factory.gpt_action_transformer.GPTActionTransformer",
            horizon=10,
            discretizer_type="binned",
        )

        target = target_from_checkpoint_dir(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="last.ckpt",
            task_label="libero",
        )

        assert target.method == "binned"
        assert target.continuous_steps is None
        assert target.trajectory_length == 10


class TestFindCheckpointDir:
    def test_finds_exact_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "gpt_conditional" / "cell__a"
        target.mkdir(parents=True)
        (tmp_path / "gpt_conditional" / "cell__a__T-120").mkdir()

        assert find_checkpoint_dir(root=tmp_path, cell_name="cell__a") == target

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_checkpoint_dir(root=tmp_path, cell_name="cell__missing")

    def test_ambiguous_directories_raise(self, tmp_path: Path) -> None:
        (tmp_path / "one" / "cell__a").mkdir(parents=True)
        (tmp_path / "two" / "cell__a").mkdir(parents=True)

        with pytest.raises(ValueError, match="Ambiguous"):
            find_checkpoint_dir(root=tmp_path, cell_name="cell__a")
