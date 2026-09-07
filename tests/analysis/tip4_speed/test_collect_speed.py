"""Tests for the Tip 4 latency aggregation."""

import csv
from pathlib import Path

from versatil.analysis.tip4_speed.collect import collect, executed_steps

SPEED_FIELDS = (
    "cell",
    "task",
    "method",
    "sigma_multiplier",
    "data_seed",
    "train_seed",
    "trajectory_length",
    "device",
    "device_name",
    "precision",
    "param_count",
    "chunk_index",
    "sequential_steps",
    "generated_tokens",
    "t_pre_ms",
    "t_encode_ms",
    "t_generate_ms",
    "t_detok_ms",
    "t_unnorm_ms",
    "t_total_ms",
    "detok_warnings",
    "weights",
)


def _speed_row(
    cell: str,
    method: str,
    chunk_index: int,
    t_generate_ms: float,
    t_total_ms: float,
    steps: int,
) -> dict[str, object]:
    return {
        "cell": cell,
        "task": "conditional",
        "method": method,
        "sigma_multiplier": 1.0,
        "data_seed": 42,
        "train_seed": 0,
        "trajectory_length": 60,
        "device": "cuda",
        "device_name": "H100",
        "precision": "32",
        "param_count": 1000,
        "chunk_index": chunk_index,
        "sequential_steps": steps,
        "generated_tokens": steps if method in ("fast", "binned") else "",
        "t_pre_ms": 1.0,
        "t_encode_ms": 2.0,
        "t_generate_ms": t_generate_ms,
        "t_detok_ms": 0.5,
        "t_unnorm_ms": 0.1,
        "t_total_ms": t_total_ms,
        "detok_warnings": 0,
        "weights": "trained",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(SPEED_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


class TestExecutedSteps:
    def test_tokenized_methods_execute_one_less_than_trajectory(self) -> None:
        assert executed_steps(method="fast", trajectory_length=60) == 59
        assert executed_steps(method="binned", trajectory_length=240) == 239

    def test_continuous_methods_execute_the_full_trajectory(self) -> None:
        assert executed_steps(method="qfat", trajectory_length=60) == 60
        assert executed_steps(method="bcat", trajectory_length=60) == 60


class TestCollect:
    def test_aggregates_medians_shares_and_success_join(self, tmp_path: Path) -> None:
        rows = [
            _speed_row("cell_fast_s0", "fast", 0, 10.0, 20.0, 40),
            _speed_row("cell_fast_s0", "fast", 1, 30.0, 40.0, 50),
            _speed_row("cell_fast_s0", "fast", 2, 20.0, 30.0, 45),
        ]
        speed_csv = tmp_path / "results_speed_cuda.csv"
        _write_csv(speed_csv, rows)
        tip1_csv = tmp_path / "results_all.csv"
        with open(tip1_csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["cell", "final_conditional_success"]
            )
            writer.writeheader()
            writer.writerow({"cell": "cell_fast_s0", "final_conditional_success": 0.95})

        output_csv = collect(
            speed_csv=speed_csv, output_dir=tmp_path, tip1_results=tip1_csv
        )

        with open(output_csv, newline="") as csv_file:
            summary = list(csv.DictReader(csv_file))
        assert len(summary) == 1
        row = summary[0]
        assert row["method"] == "fast"
        assert float(row["t_generate_ms"]) == 20.0  # median of 10/30/20
        assert float(row["t_total_ms_mean"]) == 30.0
        assert float(row["t_generate_share"]) == 0.6667  # median of 0.5/0.75/0.667
        assert float(row["steps_mean"]) == 45.0
        assert float(row["success_mean"]) == 0.95
        # 59 executed steps / 30 ms per chunk.
        assert float(row["control_rate_hz"]) == round(59 / 0.030, 1)

    def test_seed_replicates_average_with_error_bar(self, tmp_path: Path) -> None:
        rows = [
            _speed_row("cell_qfat_s0", "qfat", 0, 10.0, 20.0, 60),
            _speed_row("cell_qfat_s1", "qfat", 0, 10.0, 40.0, 60),
        ]
        speed_csv = tmp_path / "results_speed_cuda.csv"
        _write_csv(speed_csv, rows)

        output_csv = collect(
            speed_csv=speed_csv, output_dir=tmp_path, tip1_results=None
        )

        with open(output_csv, newline="") as csv_file:
            summary = list(csv.DictReader(csv_file))
        assert len(summary) == 1
        row = summary[0]
        assert int(row["num_cells"]) == 2
        assert float(row["t_total_ms_mean"]) == 30.0
        assert float(row["t_total_ms_sd"]) > 0.0
        assert row["success_mean"] == ""
