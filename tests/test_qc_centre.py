# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""The quantum-chemistry tiers compute the central step of each context."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "qc"))
B = pytest.importorskip("build_qc_fragments", reason="physics environment not active")


def test_central_step_is_centred():
    assert B.central_step_index("ACGT") == 1            # tetramer: unchanged
    assert B.central_step_index("AACGTT") == 2          # hexamer: two bases either side
    for ctx in ("ACGT", "AACGTT", "AAACGTTT"):
        i = B.central_step_index(ctx)
        assert i == len(ctx) - (i + 2)                  # as many bases left as right
    with pytest.raises(ValueError):
        B.central_step_index("ACG")
