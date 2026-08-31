"""Tests for versatil.analysis.tip1_noise.collect module."""

from collections.abc import Callable
from pathlib import Path

import pytest

from versatil.analysis.tip1_noise.collect import NAME_PATTERN, parse_log, summarize

CELL_NAME = "conditional__inj-position__band-high__sig-2__dseed-42__fast__seed-0"
COMMON_FIELDS = (
    "collision=0.10, endpoint_reach=1.00, path_length=1.00, "
    "valid_mode_coverage=0.50, valid_entropy={entropy}, "
    "raw_mode_coverage=0.50, raw_entropy=0.40, per_mode={{0: 5, 1: 5}}"
)


def _rollout_line(
    epoch: int,
    success: float,
    entropy: str = "0.40",
    context_accuracy: float | None = None,
    conditional_success: float | None = None,
) -> str:
    line = (
        f"2026-08-27 10:00:00,000 synthetic_rollout INFO Synthetic rollout: "
        f"epoch {epoch}, success={success:.2f}, "
        + COMMON_FIELDS.format(entropy=entropy)
    )
    if context_accuracy is not None:
        line += (
            f", context_accuracy={context_accuracy:.2f}, "
            f"conditional_success={conditional_success:.2f}"
        )
    return line


@pytest.fixture
def log_file_factory(tmp_path: Path) -> Callable[..., Path]:
    def factory(lines: list[str], finished: bool = True) -> Path:
        content = [f"[1/16] {CELL_NAME}", *lines]
        if finished:
            content.append("Finished at: Wed Aug 27 10:00:00 CEST 2026")
        path = tmp_path / "tip1_train_1_0.log"
        path.write_text("\n".join(content) + "\n")
        return path

    return factory


@pytest.mark.unit
def test_parse_log_keeps_negative_entropy_evaluations(
    log_file_factory: Callable[..., Path],
):
    path = log_file_factory(
        [
            _rollout_line(epoch=0, success=0.0),
            _rollout_line(epoch=100, success=0.75, entropy="-0.00"),
        ]
    )

    result = parse_log(path)

    assert result.cell == CELL_NAME
    assert result.finished is True
    assert [item["epoch"] for item in result.evaluations] == [0.0, 100.0]
    assert result.evaluations[-1]["success"] == pytest.approx(0.75)


@pytest.mark.unit
@pytest.mark.parametrize(
    "with_context, expected_accuracy, expected_conditional",
    [
        (True, 0.95, 0.8),
        (False, "", ""),
    ],
)
def test_summarize_reports_context_fields_only_when_logged(
    log_file_factory: Callable[..., Path],
    with_context: bool,
    expected_accuracy: float | str,
    expected_conditional: float | str,
):
    final = (
        _rollout_line(
            epoch=100,
            success=0.9,
            context_accuracy=0.95,
            conditional_success=0.8,
        )
        if with_context
        else _rollout_line(epoch=100, success=0.9)
    )
    path = log_file_factory([_rollout_line(epoch=0, success=0.2), final])

    row = summarize(parse_log(path))

    assert row["task"] == "conditional"
    assert row["method"] == "fast"
    assert row["sigma_multiplier"] == 2.0
    assert row["final_epoch"] == 100
    assert row["final_success"] == pytest.approx(0.9)
    assert row["best_success"] == pytest.approx(0.9)
    assert row["final_context_accuracy"] == (
        pytest.approx(expected_accuracy) if with_context else expected_accuracy
    )
    assert row["final_conditional_success"] == (
        pytest.approx(expected_conditional) if with_context else expected_conditional
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "cell_name, expected_length, expected_episodes",
    [
        (
            "conditional__inj-position__band-high__sig-1__dseed-42__fast__seed-0",
            None,
            None,
        ),
        (
            "conditional__inj-position__band-high__sig-1__dseed-42__T-240__binned__seed-0",
            "240",
            None,
        ),
        (
            "conditional__inj-position__band-high__sig-1__dseed-43__ep-50__T-120__fast__seed-1",
            "120",
            "50",
        ),
    ],
)
def test_name_pattern_parses_the_trajectory_length_when_present(
    cell_name: str, expected_length: str | None, expected_episodes: str | None
):
    fields = NAME_PATTERN.match(cell_name)

    assert fields is not None
    assert fields.group("trajectory_length") == expected_length
    assert fields.group("num_episodes") == expected_episodes
    assert fields.group("method") in {"fast", "binned"}


@pytest.mark.unit
def test_summarize_defaults_the_trajectory_length_for_unsuffixed_cells(
    log_file_factory: Callable[..., Path],
):
    path = log_file_factory([_rollout_line(epoch=0, success=0.2)])

    row = summarize(parse_log(path))

    assert row["trajectory_length"] == 60


@pytest.mark.unit
def test_summarize_marks_unevaluated_runs_with_empty_fields(
    log_file_factory: Callable[..., Path],
):
    path = log_file_factory([], finished=False)

    row = summarize(parse_log(path))

    assert row["finished"] is False
    assert row["num_evaluations"] == 0
    assert row["final_success"] == ""
    assert row["final_conditional_success"] == ""
