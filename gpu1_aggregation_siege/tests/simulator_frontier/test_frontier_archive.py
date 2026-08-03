from dicode.simulator_frontier.archive_schema import FrontierArchiveEntry
from dicode.simulator_frontier.frontier_archive import FrontierArchive
from dicode.simulator_frontier.state_codec import StateBundle, StateCodec


def _entry(state_hash):
    return FrontierArchiveEntry("s1", "c1", "e1", 0, 4, "frontier", 1, 0.5, "mid", "low", "mid", "stone", {}, False, "ZERO_MEMORY", "s1", state_hash, "p1", "2026-08-03T00:00:00Z")


def test_archive_dedup_quota_and_save_load(tmp_path):
    codec = StateCodec()
    encoded = codec.encode(StateBundle({"x": 1}, 0, {}, 0, 0))
    archive = FrontierArchive(capacity=2, per_bucket_quota=1)
    assert archive.add(_entry(encoded.payload_hash), encoded)
    assert not archive.add(_entry(encoded.payload_hash), encoded)
    assert archive.validate() == []
    path = tmp_path / "archive.json"
    archive.save(path)
    loaded = FrontierArchive.load(path)
    assert len(loaded) == 1 and loaded.get("s1")[0].state_hash == encoded.payload_hash
