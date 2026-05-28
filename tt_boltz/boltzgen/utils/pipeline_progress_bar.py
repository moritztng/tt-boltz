"""Lightweight progress bar for the pipeline runner.

Replaces BoltzGen's PyTorch-Lightning TQDMProgressBar callback with a plain
tqdm wrapper driven by the inference loop in :mod:`tt_boltz.boltzgen.task.predict.predict`.
"""
import os

from tqdm import tqdm


class PipelineProgressBar:
    """Tiny progress reporter (no Lightning callback inheritance).

    Reads ``BOLTZGEN_PIPELINE_STEP`` and ``BOLTZGEN_PIPELINE_PROGRESS`` from the
    environment to label the bar — these are set by the pipeline runner when
    iterating through the steps in ``boltzgen run``.
    """

    def __init__(self):
        self._pbar = None

    @staticmethod
    def _label() -> str:
        step = os.environ.get("BOLTZGEN_PIPELINE_STEP", "")
        progress = os.environ.get("BOLTZGEN_PIPELINE_PROGRESS", "")
        return f"[{progress}] {step}".strip() if progress else (f"[{step}]" if step else "")

    def on_start(self, total: int = None) -> None:
        self._pbar = tqdm(total=total, desc=self._label(), leave=True)

    def on_batch_end(self, batch_idx: int = 0) -> None:
        if self._pbar is not None:
            self._pbar.update(1)

    def on_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
