"""
MLXQ Format Specification — Constants and Validation
Created by Eric Jang (eric@vmlx.net)
"""

FORMAT_NAME = "mxq"
FORMAT_VERSION = "1.0"
DEFAULT_BLOCK_SIZE = 64
ALLOWED_BIT_WIDTHS = frozenset({2, 3, 4, 5, 6, 8})
MXQ_CONFIG_FILENAME = "mxq_config.json"
MXQ_IMATRIX_FILENAME = "mxq_imatrix.safetensors"
MXQ_INDEX_FILENAME = "model.mxq.index.json"

# Bytes per block at each bit width (for block_size=64)
BYTES_PER_BLOCK = {
    2: 16,   # 64 * 2 / 8
    3: 24,   # 64 * 3 / 8
    4: 32,   # 64 * 4 / 8
    5: 40,   # 64 * 5 / 8
    6: 48,   # 64 * 6 / 8
    8: 64,   # 64 * 8 / 8
}

# Tensor name suffixes for MXQ companion tensors
QWEIGHT_SUFFIX = ".qweight"
SCALES_SUFFIX = ".scales"
ZEROS_SUFFIX = ".zeros"
BITMAP_SUFFIX = ".bit_map"
OFFSETS_SUFFIX = ".block_offsets"
IMPORTANCE_SUFFIX = ".importance"
ACT_NORMS_SUFFIX = ".act_norms"

MXQ_SUFFIXES = (QWEIGHT_SUFFIX, SCALES_SUFFIX, ZEROS_SUFFIX, BITMAP_SUFFIX, OFFSETS_SUFFIX)


def bytes_per_block(bits: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Calculate bytes needed to store one block at the given bit width."""
    if bits not in ALLOWED_BIT_WIDTHS:
        raise ValueError(f"Bit width {bits} not in {sorted(ALLOWED_BIT_WIDTHS)}")
    # Ceiling division: (block_size * bits + 7) // 8
    return (block_size * bits + 7) // 8


def effective_bits(nominal_bits: float, block_size: int = DEFAULT_BLOCK_SIZE) -> float:
    """
    Calculate effective bits per weight including metadata overhead.

    Overhead per block:
    - scale: 2 bytes (float16)
    - zero: 2 bytes (float16)
    - bit_map: 1 byte (uint8)
    - block_offset: 4 bytes (uint32)
    Total overhead: 9 bytes = 72 bits per block
    """
    overhead_bits = 72  # 9 bytes per block
    return nominal_bits + overhead_bits / block_size


def validate_bit_width(bits: int) -> None:
    """Raise ValueError if bit width is not allowed."""
    if bits not in ALLOWED_BIT_WIDTHS:
        raise ValueError(
            f"Invalid bit width {bits}. Allowed: {sorted(ALLOWED_BIT_WIDTHS)}"
        )


def compute_block_offsets(bit_map: list[int], block_size: int = DEFAULT_BLOCK_SIZE) -> list[int]:
    """Compute byte offsets for each block given their bit widths."""
    offsets = []
    current_offset = 0
    for bits in bit_map:
        offsets.append(current_offset)
        current_offset += bytes_per_block(bits, block_size)
    return offsets


def estimate_model_size(
    num_params: int,
    target_bits: float,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict:
    """Estimate MXQ model size in bytes."""
    eff_bits = effective_bits(target_bits, block_size)
    weight_bytes = int(num_params * eff_bits / 8)
    return {
        "nominal_bits": target_bits,
        "effective_bits": round(eff_bits, 2),
        "weight_bytes": weight_bytes,
        "weight_gb": round(weight_bytes / (1024 ** 3), 2),
    }
