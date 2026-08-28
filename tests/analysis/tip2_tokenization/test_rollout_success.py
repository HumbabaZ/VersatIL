"""Tests for versatil.analysis.tip2_tokenization.rollout_success module."""

import os
from pathlib import Path

import pytest

from versatil.analysis.tip2_tokenization.rollout_success import (
    final_success_by_cell,
)


@pytest.fixture
def log_factory(tmp_path: Path):
    def factory(
        filename: str,
        cell_name: str,
        successes: list[float],
        modified: float,
    ) -> Path:
        lines = [
            "workspace INFO Workspace initialized for experiment: "
            f"end_to_end_training_runs/synthetic/gpt_transformer/{cell_name}"
        ]
        lines += [
            f"synthetic_rollout INFO Synthetic rollout: epoch {epoch}, "
            f"success={value:.2f}, collision=0.00"
            for epoch, value in enumerate(successes)
        ]
        path = tmp_path / filename
        path.write_text("\n".join(lines))
        os.utime(path, (modified, modified))
        return path

    return factory


class TestFinalSuccessByCell:
    @pytest.mark.unit
    def test_takes_the_last_logged_rollout(self, tmp_path: Path, log_factory):
        log_factory(
            filename="tip2_train_1_0.log",
            cell_name="cond__fast__scale-1__seed-0",
            successes=[0.1, 0.5, 0.9],
            modified=100.0,
        )

        assert final_success_by_cell(log_dir=tmp_path) == {
            "cond__fast__scale-1__seed-0": 0.9
        }

    @pytest.mark.unit
    def test_newest_log_wins_for_a_rerun_cell(self, tmp_path: Path, log_factory):
        log_factory(
            filename="tip2_train_1_0.log",
            cell_name="cell",
            successes=[0.2],
            modified=100.0,
        )
        log_factory(
            filename="tip2_train_2_0.log",
            cell_name="cell",
            successes=[0.8],
            modified=200.0,
        )

        assert final_success_by_cell(log_dir=tmp_path) == {"cell": 0.8}

    @pytest.mark.unit
    def test_log_without_a_rollout_line_is_skipped(self, tmp_path: Path):
        (tmp_path / "tip2_train_3_0.log").write_text(
            "Workspace initialized for experiment: cfg/cell\nno rollout"
        )

        assert final_success_by_cell(log_dir=tmp_path) == {}
