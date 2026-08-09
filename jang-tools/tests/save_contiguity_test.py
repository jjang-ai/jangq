"""Non-contiguous views must not reach ``safetensors.save_file``.

safetensors serialises an array's underlying buffer and ignores numpy strides. A non-contiguous
view is therefore written as whatever bytes follow its start offset, producing a tensor with the
right name, dtype and shape and silently wrong contents.

This is not hypothetical: splitting a fused ``experts.gate_up_proj`` with ``mlx_weight[:, :mid, :]``
produced 6-bit MoE bundles in which only expert 0 was correct, out of 256. The model loaded,
generated fluent text, and scored at chance.
"""

import numpy as np
import pytest

from jang_tools.convert import _contiguous_for_save

save_file = pytest.importorskip("safetensors.numpy").save_file
load_file = pytest.importorskip("safetensors.numpy").load_file


def _fused_expert_stack():
    """A miniature of the real case: ``[n_experts, 2 * intermediate, packed]``."""
    n_experts, two_inter, packed = 4, 6, 2
    stack = np.arange(n_experts * two_inter * packed, dtype=np.uint32)
    return stack.reshape(n_experts, two_inter, packed), two_inter // 2


def test_axis1_slice_of_expert_stack_is_not_contiguous():
    """The premise. If numpy ever made these contiguous, the rest of this file is moot."""
    stack, mid = _fused_expert_stack()
    assert not stack[:, :mid, :].flags["C_CONTIGUOUS"]
    assert not stack[:, mid:, :].flags["C_CONTIGUOUS"]
    # Leading-axis slices DO stay contiguous — which is why only the 3D split was ever affected.
    flat = stack.reshape(-1, stack.shape[-1])
    assert flat[: len(flat) // 2].flags["C_CONTIGUOUS"]


def test_saving_a_raw_view_corrupts_it(tmp_path):
    """Characterises the bug itself, so a future safetensors fix makes this fail loudly."""
    stack, mid = _fused_expert_stack()
    gate = stack[:, :mid, :]
    path = tmp_path / "raw_view.safetensors"
    save_file({"gate": gate}, str(path))
    written = load_file(str(path))["gate"]

    assert written.shape == gate.shape  # shape survives, which is what makes it so quiet
    assert not np.array_equal(written, gate)
    # The written bytes are the parent buffer read linearly from the view's start offset.
    flat = stack.reshape(-1, stack.shape[-1])
    assert np.array_equal(written, flat[: stack.shape[0] * mid].reshape(gate.shape))


def test_contiguous_for_save_round_trips_a_fused_split(tmp_path):
    """The fix: both halves of a fused gate/up split survive a save/load round trip."""
    stack, mid = _fused_expert_stack()
    gate, up = stack[:, :mid, :], stack[:, mid:, :]

    path = tmp_path / "fixed.safetensors"
    save_file(_contiguous_for_save({"gate": gate, "up": up}), str(path))
    written = load_file(str(path))

    assert np.array_equal(written["gate"], gate)
    assert np.array_equal(written["up"], up)
    # Every expert, not just expert 0 — expert 0 round-trips even unfixed, because its view starts
    # at buffer offset 0. A guard that only checks expert 0 passes on a fully broken bundle.
    for expert in range(1, stack.shape[0]):
        assert np.array_equal(written["gate"][expert], stack[expert, :mid, :])
        assert np.array_equal(written["up"][expert], stack[expert, mid:, :])


def test_contiguous_for_save_does_not_copy_contiguous_arrays():
    """It runs over every tensor of every shard, so it must be free when there is nothing to do."""
    already = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assert _contiguous_for_save({"w": already})["w"] is already
