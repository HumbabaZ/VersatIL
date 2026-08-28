"""Tests for versatil.analysis.tip2_tokenization.collect_eval module."""

from pathlib import Path

import pytest

from versatil.analysis.tip2_tokenization.collect_eval import (
    CHECKPOINT_DIR_ENV,
    checkpoint_root,
    find_checkpoint_dir,
)
from versatil.analysis.tip2_tokenization.sweep import stage_cells


class TestCheckpointRoot:
    @pytest.mark.unit
    def test_appends_synthetic_subdir_to_the_env_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(CHECKPOINT_DIR_ENV, "/tmp/ckpt")
        assert checkpoint_root() == Path("/tmp/ckpt/synthetic")

    @pytest.mark.unit
    def test_unset_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(CHECKPOINT_DIR_ENV, raising=False)
        with pytest.raises(ValueError, match=CHECKPOINT_DIR_ENV):
            checkpoint_root()


class TestFindCheckpointDir:
    @pytest.mark.unit
    def test_finds_the_named_dir_below_a_config_subdir(self, tmp_path: Path):
        cell_name = "sequential__fast__scale-8p84366__seed-0"
        target = tmp_path / "gpt_transformer" / cell_name
        target.mkdir(parents=True)
        assert find_checkpoint_dir(root=tmp_path, cell_name=cell_name) == str(target)

    @pytest.mark.unit
    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="absent-cell"):
            find_checkpoint_dir(root=tmp_path, cell_name="absent-cell")


@pytest.mark.unit
def test_every_pilot_cell_name_is_findable_when_present(tmp_path: Path):
    cells = stage_cells(stage="pilot", task="sequential")
    for cell in cells:
        (tmp_path / "gpt_transformer" / cell.name).mkdir(parents=True)
    for cell in cells:
        found = find_checkpoint_dir(root=tmp_path, cell_name=cell.name)
        assert Path(found).name == cell.name
