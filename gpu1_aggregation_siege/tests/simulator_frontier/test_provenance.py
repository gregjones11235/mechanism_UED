import pytest

from dicode.simulator_frontier.errors import ProvenanceViolationError
from dicode.simulator_frontier.provenance import DataSource, FormalDataLeakageGuard, SearchActionLeakageGuard


def test_formal_source_rejected_from_training_consumers():
    with pytest.raises(ProvenanceViolationError):
        FormalDataLeakageGuard.assert_allowed(DataSource.FORMAL_FULL, "curriculum")
    FormalDataLeakageGuard.assert_allowed(DataSource.TRAINING_FRONTIER_CAPTURE, "FrontierArchive")


def test_nested_action_guidance_rejected_but_aggregate_allowed():
    with pytest.raises(ProvenanceViolationError):
        SearchActionLeakageGuard.validate_aggregate({"stats": {"route": [1, 2]}})
    SearchActionLeakageGuard.validate_aggregate({"success": True, "progress": 0.4, "cost": 8})
