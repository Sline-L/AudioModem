import numpy as np
import pytest

from n3_2.ldpc_codec import CODE_BITS, INFO_BITS, StandardLdpc
from n3_2.modem import scramble_bits


def test_noiseless_ldpc_roundtrip_and_syndrome():
    codec = StandardLdpc()
    info = np.random.default_rng(12).integers(0, 2, INFO_BITS, dtype=np.uint8)
    coded = codec.encode(info)
    assert coded.shape == (CODE_BITS,)
    assert coded.dtype == np.uint8
    assert not codec.syndrome(coded).any()
    decoded, ok = codec.decode_llr((1 - 2 * coded.astype(float)) * 8)
    assert ok and np.array_equal(decoded, info)


def test_soft_descrambling_precedes_decode():
    codec = StandardLdpc()
    info = np.zeros(INFO_BITS, np.uint8)
    coded = codec.encode(info)
    decoded, ok = codec.decode_llr((1 - 2 * scramble_bits(coded).astype(float)) * 6)
    assert ok and np.array_equal(decoded, info)


def test_ldpc_inputs_require_binary_flat_information_and_full_llrs():
    codec = StandardLdpc()
    with pytest.raises(ValueError, match="1992"):
        codec.encode(np.zeros(INFO_BITS - 1, dtype=np.uint8))
    with pytest.raises(ValueError, match="zero or one"):
        codec.encode(np.full(INFO_BITS, 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="3984"):
        codec.decode_llr(np.zeros(CODE_BITS - 1))
