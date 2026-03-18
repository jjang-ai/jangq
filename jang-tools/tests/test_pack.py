"""Tests for JANG bit packing/unpacking."""

import numpy as np
import pytest
from jang_tools.pack import pack_bits, unpack_bits, pack_block, unpack_block


class TestPackUnpack:
    """Verify pack/unpack roundtrip for all supported bit widths."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_roundtrip(self, bits):
        """Pack then unpack should recover original values."""
        n = 64  # one block
        max_val = (1 << bits) - 1
        rng = np.random.default_rng(42)
        values = rng.integers(0, max_val + 1, size=n, dtype=np.uint8)

        packed = pack_bits(values, bits)
        unpacked = unpack_bits(packed, bits, n)

        np.testing.assert_array_equal(unpacked, values)

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_packed_size(self, bits):
        """Packed array should have the expected number of bytes."""
        n = 64
        values = np.zeros(n, dtype=np.uint8)
        packed = pack_bits(values, bits)
        expected_bytes = (n * bits + 7) // 8
        assert len(packed) == expected_bytes

    def test_2bit_specific(self):
        """Test 2-bit packing with known values."""
        values = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        packed = pack_bits(values, 2)
        # 00 01 10 11 | 00 01 10 11
        # = 0b11100100 | 0b11100100
        # = 0xE4       | 0xE4
        assert packed[0] == 0xE4
        assert packed[1] == 0xE4

        unpacked = unpack_bits(packed, 2, 8)
        np.testing.assert_array_equal(unpacked, values)

    def test_4bit_specific(self):
        """Test 4-bit packing with known values."""
        values = np.array([5, 10, 3, 15], dtype=np.uint8)
        packed = pack_bits(values, 4)
        # byte 0: low=5, high=10 → 0xA5
        # byte 1: low=3, high=15 → 0xF3
        assert packed[0] == 0xA5
        assert packed[1] == 0xF3

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_block_roundtrip(self, bits):
        """Test pack_block/unpack_block convenience functions."""
        block_size = 64
        max_val = (1 << bits) - 1
        rng = np.random.default_rng(123)
        values = rng.integers(0, max_val + 1, size=block_size, dtype=np.uint8)

        packed = pack_block(values, bits)
        unpacked = unpack_block(packed, bits, block_size)

        np.testing.assert_array_equal(unpacked, values)

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_large_roundtrip(self, bits):
        """Test with many blocks worth of data."""
        n = 64 * 100  # 100 blocks
        max_val = (1 << bits) - 1
        rng = np.random.default_rng(999)
        values = rng.integers(0, max_val + 1, size=n, dtype=np.uint8)

        packed = pack_bits(values, bits)
        unpacked = unpack_bits(packed, bits, n)

        np.testing.assert_array_equal(unpacked, values)

    def test_max_values(self):
        """Test that maximum values at each bit width pack correctly."""
        for bits in [2, 3, 4, 5, 6, 8]:
            max_val = (1 << bits) - 1
            values = np.full(64, max_val, dtype=np.uint8)
            packed = pack_bits(values, bits)
            unpacked = unpack_bits(packed, bits, 64)
            np.testing.assert_array_equal(unpacked, values)

    def test_zero_values(self):
        """Test that all-zero arrays pack correctly."""
        for bits in [2, 3, 4, 5, 6, 8]:
            values = np.zeros(64, dtype=np.uint8)
            packed = pack_bits(values, bits)
            unpacked = unpack_bits(packed, bits, 64)
            np.testing.assert_array_equal(unpacked, values)
