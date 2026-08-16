from dicode.e3_litesim.measurement.causal_evidence import (CauseRecord,
                                                           aggregate_causal_evidence)
from dicode.e3_litesim.scheduler.deterministic_scheduler import (
    DeterministicScheduler)


class _F:
    spec_hash = "abc"


def test_deterministic_and_evidence_shift():
    sched = DeterministicScheduler()
    d1 = sched.build_distribution(_F(), None)
    d2 = sched.build_distribution(_F(), None)
    assert d1 == d2
    ev = aggregate_causal_evidence([
        CauseRecord("visibility_plus", "PERCEPTION", 0.5, 0.1, {})])
    d3 = sched.build_distribution(_F(), ev)
    assert d3["weights"]["repair"] > d1["weights"]["repair"]
    assert abs(sum(d3["weights"].values()) - 1.0) < 1e-9