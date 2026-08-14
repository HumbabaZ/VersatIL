"""Configuration for the FAST rate-distortion sweep."""

from dataclasses import dataclass, field


@dataclass
class FastRateDistortionConfig:
    """Grid and data settings for the FAST tokenizer rate-distortion sweep.

    The sweep is a sparse cross (spec section 4): the rounding-scale sweep holds
    ``scale_sweep_fixed_vocab`` fixed while varying ``scales``; the vocabulary
    sweep holds ``vocab_sweep_fixed_scale`` fixed while varying ``vocab_sizes``.
    The two sweeps share the ``(vocab_sweep_fixed_scale,
    scale_sweep_fixed_vocab)`` operating point, which is the main-experiment
    training configuration.

    Attributes:
        scales: DCT rounding scales for the distortion sweep. Larger means finer
            quantization and lower distortion.
        vocab_sizes: BPE vocabulary sizes for the rate sweep. Distortion is
            expected to stay flat across these.
        scale_sweep_fixed_vocab: Vocabulary size held fixed during the scale
            sweep.
        vocab_sweep_fixed_scale: Rounding scale held fixed during the vocabulary
            sweep.
        near_lossless_scale: Fine-quantization scale used for the sanity check.
        near_lossless_vocab: Large vocabulary used for the sanity check.
        near_lossless_tolerance: Maximum per-dimension original-space RMSE the
            near-lossless round trip may reach before the sweep is trusted.
        max_chunks: Cap on the number of action chunks used, for a bounded
            runtime. Set to 0 to use every chunk.
        pretrained_fast_model: HuggingFace model id or local path whose remote
            code supplies the ``UniversalActionProcessor`` class used for fitting.
        families: Tokenizer families to sweep (``fast`` and/or ``binning``).
        bin_counts: Per-value binning bin counts for the binning baseline (G3).
        binning_strategy: Binning edge strategy (``quantile`` or ``uniform``).
        horizons: Chunk horizons to sweep (G4). The FAST/binning knob sweeps run
            at the operating point for every horizon; horizon 10 is the main cell.
    """

    scales: list[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 20.0, 50.0])
    vocab_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    scale_sweep_fixed_vocab: int = 1024
    vocab_sweep_fixed_scale: float = 10.0
    near_lossless_scale: float = 100.0
    near_lossless_vocab: int = 4096
    near_lossless_tolerance: float = 0.05
    max_chunks: int = 20000
    pretrained_fast_model: str = "physical-intelligence/fast"
    families: list[str] = field(default_factory=lambda: ["fast", "binning"])
    bin_counts: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024])
    binning_strategy: str = "quantile"
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
