"""JANG affine storage helpers, including the native 1-bit extension.

JANG affine tensors use the same equation at every supported storage width::

    dequantized = packed_code * scale + bias

MLX 0.31 supports 2/3/4/5/6/8-bit affine kernels but not 1-bit storage.
The 1-bit JANG extension therefore remains 1-bit on disk and expands its
codes from ``{0, 1}`` to the identical 2-bit codes ``{0, 1}`` at load time.
Scales and biases are unchanged, so the runtime expansion is bit-exact and
does not introduce a second quantization step.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .pack import pack_bits, unpack_bits


AFFINE1_RUNTIME_BITS = 2
AFFINE1_STORAGE_BITS = 1
SUPPORTED_AFFINE_GROUP_SIZES = frozenset({32, 64, 128})


def _validate_matrix_shape(weights: np.ndarray, group_size: int) -> tuple[int, int]:
    if weights.ndim != 2:
        raise ValueError(f"affine quantization expects a 2D matrix, got {weights.shape}")
    rows, columns = (int(weights.shape[0]), int(weights.shape[1]))
    if group_size not in SUPPORTED_AFFINE_GROUP_SIZES:
        raise ValueError(
            f"unsupported affine group_size={group_size}; "
            f"expected one of {sorted(SUPPORTED_AFFINE_GROUP_SIZES)}"
        )
    if columns % group_size:
        raise ValueError(
            f"input dimension {columns} is not divisible by group_size={group_size}"
        )
    return rows, columns


def quantize_discrete_affine(
    weights: np.ndarray,
    *,
    bits: int,
    group_size: int = 128,
    chunk_rows: int = 256,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Encode exact binary or ternary groupwise weights as JANG affine.

    ``bits=1`` expects at most two values per group and assigns the lower and
    upper values to codes 0 and 1. ``bits=2`` expects at most three evenly
    spaced values per group and assigns them to codes 0, 1 and 2; code 3 stays
    unused. Constant and two-level groups are also represented exactly.

    The returned arrays match MLX/JANG v2 tensor conventions:

    - packed weight: ``uint32[rows, columns * bits / 32]``
    - scales: ``float16[rows, columns / group_size]``
    - biases: ``float16[rows, columns / group_size]``

    The final two return values report relative-L1 and maximum absolute error
    introduced only by standard float16 scale/bias storage. Validation still
    requires the source levels to be exactly affine-representable before that
    metadata rounding.
    """
    if bits not in (1, 2):
        raise ValueError(f"discrete affine only supports bits=1 or bits=2, got {bits}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

    matrix = np.asarray(weights, dtype=np.float32)
    rows, columns = _validate_matrix_shape(matrix, group_size)
    groups_per_row = columns // group_size
    packed_columns = columns * bits // 32
    if columns * bits % 32:
        raise ValueError(
            f"packed row is not uint32-aligned: columns={columns}, bits={bits}"
        )

    packed = np.empty((rows, packed_columns), dtype=np.uint32)
    scales = np.empty((rows, groups_per_row), dtype=np.float16)
    biases = np.empty((rows, groups_per_row), dtype=np.float16)
    max_code = (1 << bits) - 1
    error_sum = 0.0
    source_sum = 0.0
    max_abs_error = 0.0

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        grouped = matrix[start:end].reshape(-1, groups_per_row, group_size)
        low = grouped.min(axis=-1)
        high = grouped.max(axis=-1)

        if bits == 1:
            scale = high - low
        else:
            # Three-level Bonsai groups are {-s, 0, +s}. The half-range is
            # therefore the exact affine step and leaves code 3 unused.
            scale = (high - low) * 0.5

        constant = scale == 0
        safe_scale = np.where(constant, 1.0, scale)
        codes = np.rint((grouped - low[..., None]) / safe_scale[..., None])
        codes = np.clip(codes, 0, max_code).astype(np.uint8)
        if np.any(constant):
            codes[constant] = 0

        if validate:
            ideal_reconstructed = (
                codes.astype(np.float32)
                * np.where(constant, 1.0, scale)[..., None].astype(np.float32)
                + low[..., None].astype(np.float32)
            )
            if not np.array_equal(ideal_reconstructed, grouped):
                abs_error = np.abs(ideal_reconstructed - grouped)
                max_error = float(abs_error.max(initial=0.0))
                bad_group = np.argwhere(np.any(abs_error != 0, axis=-1))[0]
                global_row = start + int(bad_group[0])
                group_index = int(bad_group[1])
                unique_values = np.unique(grouped[tuple(bad_group)]).tolist()
                raise ValueError(
                    "source group is not losslessly representable by the requested "
                    f"{bits}-bit affine policy: row={global_row}, "
                    f"group={group_index}, unique_values={unique_values[:8]}, "
                    f"max_abs_error={max_error}"
                )

        scale_f16 = np.where(constant, 1.0, scale).astype(np.float16)
        bias_f16 = low.astype(np.float16)
        stored_reconstructed = (
            codes.astype(np.float32) * scale_f16[..., None].astype(np.float32)
            + bias_f16[..., None].astype(np.float32)
        )
        stored_error = np.abs(stored_reconstructed - grouped)
        error_sum += float(stored_error.sum(dtype=np.float64))
        source_sum += float(np.abs(grouped).sum(dtype=np.float64))
        max_abs_error = max(
            max_abs_error, float(stored_error.max(initial=0.0))
        )

        packed_bytes = pack_bits(codes.reshape(-1), bits)
        expected_bytes = (end - start) * packed_columns * 4
        if packed_bytes.nbytes != expected_bytes:
            raise RuntimeError(
                f"packed byte count mismatch: got {packed_bytes.nbytes}, "
                f"expected {expected_bytes}"
            )
        packed[start:end] = packed_bytes.view(np.uint32).reshape(
            end - start, packed_columns
        )
        scales[start:end] = scale_f16
        biases[start:end] = bias_f16

    return (
        packed,
        scales,
        biases,
        error_sum / max(source_sum, 1e-12),
        max_abs_error,
    )


def quantize_native_affine_numpy(
    weights: np.ndarray,
    *,
    bits: int,
    group_size: int,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """CPU implementation of MLX native affine quantization.

    This mirrors MLX's ``affine_quantize`` kernel, including its signed scale,
    dominant edge selection, round-to-nearest-even behavior, and float16
    source/sidecar types. It lets large converters produce MLX-native affine
    tensors without scheduling Metal work alongside a running model server.
    """
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported native affine bits={bits}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

    matrix = np.asarray(weights, dtype=np.float16)
    rows, columns = _validate_matrix_shape(matrix, group_size)
    if columns * bits % 32:
        raise ValueError(
            f"packed row is not uint32-aligned: columns={columns}, bits={bits}"
        )
    groups_per_row = columns // group_size
    packed_columns = columns * bits // 32
    packed = np.empty((rows, packed_columns), dtype=np.uint32)
    scales = np.empty((rows, groups_per_row), dtype=np.float16)
    biases = np.empty((rows, groups_per_row), dtype=np.float16)
    bins = np.float32((1 << bits) - 1)
    error_sum = 0.0
    source_sum = 0.0

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        grouped = matrix[start:end].astype(np.float32).reshape(
            -1, groups_per_row, group_size
        )
        low = grouped.min(axis=-1)
        # MLX initializes the maximum reduction to zero, so an all-negative
        # group keeps zero as the upper edge.
        high = np.maximum(grouped.max(axis=-1), np.float32(0.0))
        scale = np.maximum((high - low) / bins, np.float32(1e-7))
        dominant_low = np.abs(low) > np.abs(high)
        scale = np.where(dominant_low, scale, -scale)
        edge = np.where(dominant_low, low, high)
        edge_code = np.rint(edge / scale)
        at_zero = edge_code == 0
        refined_scale = np.divide(
            edge,
            edge_code,
            out=scale.copy(),
            where=~at_zero,
        )
        scale = np.where(at_zero, scale, refined_scale)
        bias = np.where(at_zero, np.float32(0.0), edge)
        codes = np.minimum(
            np.rint((grouped - bias[..., None]) / scale[..., None]), bins
        ).astype(np.uint8)

        scale_f16 = scale.astype(np.float16)
        bias_f16 = bias.astype(np.float16)
        packed_bytes = pack_bits(codes.reshape(-1), bits)
        expected_bytes = (end - start) * packed_columns * 4
        if packed_bytes.nbytes != expected_bytes:
            raise RuntimeError(
                f"packed byte count mismatch: got {packed_bytes.nbytes}, "
                f"expected {expected_bytes}"
            )
        packed[start:end] = packed_bytes.view(np.uint32).reshape(
            end - start, packed_columns
        )
        scales[start:end] = scale_f16
        biases[start:end] = bias_f16

        reconstructed = (
            codes.astype(np.float32) * scale_f16[..., None].astype(np.float32)
            + bias_f16[..., None].astype(np.float32)
        )
        error_sum += float(np.abs(reconstructed - grouped).sum(dtype=np.float64))
        source_sum += float(np.abs(grouped).sum(dtype=np.float64))

    return packed, scales, biases, error_sum / max(source_sum, 1e-12)


def quantize_imatrix_affine_numpy(
    weights: np.ndarray,
    importance: np.ndarray,
    *,
    bits: int,
    group_size: int,
    chunk_rows: int = 64,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Activation-weighted affine fit with the native MLX storage ABI.

    ``importance[j]`` is proportional to ``E[x_j**2]`` for the input consumed
    by the matrix. Codes, fp16 scales, and fp16 biases are emitted in exactly
    the layout consumed by ``mx.quantized_matmul``/``mx.gather_qmm``. Unlike a
    codebook codec, this needs no runtime kernel or sidecar.
    """
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported imatrix affine bits={bits}")
    if chunk_rows <= 0 or iterations <= 0:
        raise ValueError("chunk_rows and iterations must be positive")

    matrix = np.asarray(weights, dtype=np.float32)
    rows, columns = _validate_matrix_shape(matrix, group_size)
    if columns * bits % 32:
        raise ValueError(
            f"packed row is not uint32-aligned: columns={columns}, bits={bits}"
        )
    imp = np.asarray(importance, dtype=np.float32)
    if imp.shape != (columns,):
        raise ValueError(f"importance must have shape {(columns,)}, got {imp.shape}")
    if not np.isfinite(imp).all() or np.any(imp < 0):
        raise ValueError("importance values must be finite and nonnegative")
    positive = imp[imp > 0]
    floor = float(np.median(positive)) * 1e-4 if positive.size else 1.0
    imp = np.maximum(imp, max(floor, 1e-12))

    groups_per_row = columns // group_size
    packed_columns = columns * bits // 32
    packed = np.empty((rows, packed_columns), dtype=np.uint32)
    scales = np.empty((rows, groups_per_row), dtype=np.float16)
    biases = np.empty((rows, groups_per_row), dtype=np.float16)
    grouped_imp = imp.reshape(1, groups_per_row, group_size).astype(np.float64)
    bins = float((1 << bits) - 1)
    weighted_error = 0.0
    weighted_reference = 0.0

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        grouped = matrix[start:end].reshape(-1, groups_per_row, group_size)
        low = grouped.min(axis=-1).astype(np.float64)
        high = grouped.max(axis=-1).astype(np.float64)
        scale = np.maximum((high - low) / bins, 1e-8)
        bias = low
        codes = np.clip(
            np.rint((grouped - bias[..., None]) / scale[..., None]), 0, bins
        ).astype(np.float64)

        values = grouped.astype(np.float64)
        h = grouped_imp
        sum_h = np.sum(h, axis=-1)
        sum_hw = np.sum(h * values, axis=-1)
        for _ in range(iterations):
            sum_hq = np.sum(h * codes, axis=-1)
            sum_hq2 = np.sum(h * codes * codes, axis=-1)
            sum_hqw = np.sum(h * codes * values, axis=-1)
            determinant = sum_h * sum_hq2 - sum_hq * sum_hq
            fitted_scale = np.divide(
                sum_h * sum_hqw - sum_hq * sum_hw,
                determinant,
                out=scale.copy(),
                where=np.abs(determinant) > 1e-20,
            )
            fitted_bias = (sum_hw - fitted_scale * sum_hq) / sum_h
            valid = np.isfinite(fitted_scale) & (np.abs(fitted_scale) >= 1e-8)
            scale = np.where(valid, fitted_scale, scale)
            bias = np.where(valid, fitted_bias, bias)
            codes = np.clip(
                np.rint((values - bias[..., None]) / scale[..., None]), 0, bins
            )

        scale_f16 = scale.astype(np.float16)
        bias_f16 = bias.astype(np.float16)
        stored_scale = scale_f16.astype(np.float32)
        stored_bias = bias_f16.astype(np.float32)
        codes_u8 = np.clip(
            np.rint(
                (grouped - stored_bias[..., None]) / stored_scale[..., None]
            ),
            0,
            bins,
        ).astype(np.uint8)
        reconstructed = (
            codes_u8.astype(np.float32) * stored_scale[..., None]
            + stored_bias[..., None]
        )
        weighted_error += float(
            np.sum(grouped_imp * np.square(reconstructed - grouped), dtype=np.float64)
        )
        weighted_reference += float(
            np.sum(grouped_imp * np.square(grouped), dtype=np.float64)
        )

        packed_bytes = pack_bits(codes_u8.reshape(-1), bits)
        expected_bytes = (end - start) * packed_columns * 4
        if packed_bytes.nbytes != expected_bytes:
            raise RuntimeError(
                f"packed byte count mismatch: got {packed_bytes.nbytes}, "
                f"expected {expected_bytes}"
            )
        packed[start:end] = packed_bytes.view(np.uint32).reshape(
            end - start, packed_columns
        )
        scales[start:end] = scale_f16
        biases[start:end] = bias_f16

    return (
        packed,
        scales,
        biases,
        float(np.sqrt(weighted_error / max(weighted_reference, 1e-30))),
    )


def expand_packed_1bit_to_2bit_numpy(packed: np.ndarray) -> np.ndarray:
    """Expand uint32-packed 1-bit codes to uint32-packed 2-bit codes."""
    source = np.asarray(packed, dtype=np.uint32)
    result = np.zeros((*source.shape[:-1], source.shape[-1] * 2), dtype=np.uint32)
    low = np.zeros_like(source)
    high = np.zeros_like(source)
    for index in range(16):
        low |= ((source >> np.uint32(index)) & np.uint32(1)) << np.uint32(2 * index)
        high |= (
            ((source >> np.uint32(index + 16)) & np.uint32(1))
            << np.uint32(2 * index)
        )
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result


def expand_packed_1bit_to_2bit_mlx(packed):
    """MLX-lazy equivalent of :func:`expand_packed_1bit_to_2bit_numpy`."""
    import mlx.core as mx

    source = packed.astype(mx.uint32)
    low = mx.zeros_like(source)
    high = mx.zeros_like(source)
    for index in range(16):
        low = mx.bitwise_or(
            low,
            mx.left_shift(
                mx.bitwise_and(mx.right_shift(source, index), 1), 2 * index
            ),
        )
        high = mx.bitwise_or(
            high,
            mx.left_shift(
                mx.bitwise_and(mx.right_shift(source, index + 16), 1), 2 * index
            ),
        )
    return mx.reshape(
        mx.stack((low, high), axis=-1),
        (*source.shape[:-1], source.shape[-1] * 2),
    )


def dequantize_affine_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    bits: int,
    group_size: int,
) -> np.ndarray:
    """Reference decoder used by converter verification and focused tests."""
    packed_array = np.asarray(packed, dtype=np.uint32)
    rows = int(packed_array.shape[0])
    columns = int(packed_array.shape[-1]) * 32 // bits
    codes = unpack_bits(packed_array.view(np.uint8), bits, rows * columns)
    codes = codes.reshape(rows, columns).astype(np.float32)
    return (
        codes.reshape(rows, -1, group_size)
        * np.asarray(scales, dtype=np.float32)[..., None]
        + np.asarray(biases, dtype=np.float32)[..., None]
    ).reshape(rows, columns)


def affine1_storage_modules(jang_config: Mapping[str, Any]) -> frozenset[str]:
    """Return module paths stored with the JANG 1-bit affine extension."""
    quantization = jang_config.get("quantization")
    if not isinstance(quantization, Mapping):
        return frozenset()
    manifest = quantization.get("tensor_quantization_manifest")
    if not isinstance(manifest, Mapping):
        return frozenset()
    return frozenset(
        str(module_path)
        for module_path, spec in manifest.items()
        if isinstance(module_path, str)
        and isinstance(spec, Mapping)
        and int(spec.get("storage_bits", spec.get("bits", 0)) or 0)
        == AFFINE1_STORAGE_BITS
    )


def prepare_affine1_runtime_config(
    config: Mapping[str, Any],
    jang_config: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Rewrite only 1-bit storage specs to native 2-bit runtime specs.

    The returned config is an in-memory copy. Artifact metadata remains honest:
    ``config.json`` and ``jang_config.json`` continue to describe 1-bit storage.
    """
    modules = affine1_storage_modules(jang_config)
    runtime_config = copy.deepcopy(dict(config))
    if not modules:
        return runtime_config, modules

    quantization = runtime_config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("JANG affine-1 bundle is missing config.json quantization")

    if int(quantization.get("bits", 0) or 0) == AFFINE1_STORAGE_BITS:
        quantization["bits"] = AFFINE1_RUNTIME_BITS
    for module_path in modules:
        spec = quantization.get(module_path)
        if not isinstance(spec, dict):
            raise ValueError(
                f"JANG affine-1 module {module_path!r} is missing its config override"
            )
        storage_bits = int(spec.get("storage_bits", spec.get("bits", 0)) or 0)
        if storage_bits != AFFINE1_STORAGE_BITS:
            raise ValueError(
                f"JANG affine-1 module {module_path!r} has storage_bits={storage_bits}"
            )
        spec["storage_bits"] = AFFINE1_STORAGE_BITS
        spec["bits"] = AFFINE1_RUNTIME_BITS
        spec["mode"] = "affine"

    quantization["runtime_expansion"] = {
        "storage_bits": AFFINE1_STORAGE_BITS,
        "runtime_bits": AFFINE1_RUNTIME_BITS,
        "lossless": True,
    }
    runtime_config["quantization"] = quantization
    return runtime_config, modules


def expand_affine1_shard_mlx(
    weights: Mapping[str, Any],
    storage_modules: frozenset[str] | set[str],
) -> tuple[dict[str, Any], int]:
    """Expand every indexed 1-bit packed weight present in an MLX shard."""
    if not storage_modules:
        return dict(weights), 0
    expanded = dict(weights)
    count = 0
    for module_path in storage_modules:
        weight_key = f"{module_path}.weight"
        value = expanded.get(weight_key)
        if value is None:
            continue
        expanded[weight_key] = expand_packed_1bit_to_2bit_mlx(value)
        count += 1
    return expanded, count
