"""Open-loop replay-success distortion measure for action tokenizers.

This is the task-level tokenizer floor from the design spec: decode the tokenized
demonstration action chunks, replay them open-loop in the LIBERO simulator, and
score the standard success criterion. VersatIL has no environment integration, so
this is a stub. The LIBERO simulator is installed at
``/home/h5/qizh093f/simulation_libero`` and can drive the real implementation.
"""


def replay_success() -> float:
    """Return the open-loop replay success rate for decoded action chunks.

    Not yet implemented: returns NaN so the sweep table and figure carry the
    column while the reconstruction-error analysis proceeds.

    Returns:
        ``float('nan')`` until the LIBERO replay harness is wired in.
    """
    # TODO: implement open-loop LIBERO replay.
    #   1. Group decoded chunks back into their source demonstrations.
    #   2. For each demonstration, reset the LIBERO env from the demo's initial
    #      state (bddl task + init_state) using the installed simulator at
    #      /home/h5/qizh093f/simulation_libero.
    #   3. Step the decoded actions strictly open-loop (no re-planning).
    #   4. Read LIBERO's success flag and average over demonstrations.
    return float("nan")
