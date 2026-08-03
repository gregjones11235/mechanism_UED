import numpy as np
import pytest

from dicode.simulator_frontier.errors import SchemaMismatchError
from dicode.simulator_frontier.state_codec import StateBundle, StateCodec


def test_round_trip_shape_dtype_and_hash():
    bundle = StateBundle({"x": np.array([1, 2], dtype=np.int32)}, np.array([3], dtype=np.uint64),
                         {"flag": True}, 2, np.float32(1.5), policy_memory=(np.ones((2, 2)),), history_reference="h1")
    codec = StateCodec()
    encoded = codec.encode(bundle)
    restored = codec.decode(encoded)
    assert np.array_equal(restored.env_state["x"], bundle.env_state["x"])
    assert restored.env_state["x"].dtype == np.int32
    assert codec.encode(restored).payload_hash == encoded.payload_hash


def test_corruption_and_schema_fail_closed():
    codec = StateCodec()
    encoded = codec.encode(StateBundle({}, None, {}, None, 0))
    bad = encoded.__class__(encoded.schema_version, encoded.tree_definition, encoded.arrays,
                            encoded.scalar_metadata, encoded.payload, "0" * 64, encoded.codec_version)
    with pytest.raises(SchemaMismatchError):
        codec.decode(bad)
