from dicode.simulator_frontier.memory_modes import MemoryRestoreMode, MemoryRestoreRequest, validate_memory_request


def test_zero_memory_is_explicit_warning():
    report = validate_memory_request(MemoryRestoreRequest(MemoryRestoreMode.ZERO_MEMORY, "arch"))
    assert report.compatible and "ZERO_MEMORY_EXPLICIT_DIAGNOSTIC_ONLY" in report.warning_codes


def test_saved_memory_requires_identity_and_structure():
    req = MemoryRestoreRequest(MemoryRestoreMode.SAVED_POLICY_MEMORY, "arch", "ckpt", {"x": 1})
    assert not validate_memory_request(req, checkpoint_id="other", architecture_id="arch", memory_tree_structure={"x": 1}).compatible
    assert validate_memory_request(req, checkpoint_id="ckpt", architecture_id="arch", memory_tree_structure={"x": 1}).compatible
