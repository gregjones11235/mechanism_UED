import jax
import numpy as np

from dicode.e3_litesim.data.state_bank import FrontierStateBank
from test_state_restore_replay import _capsule
from helpers import make_setup


def test_bank_provenance_and_validity():
    s = make_setup()
    cap = _capsule(s)
    bank = FrontierStateBank("BASIC_SURVIVAL")
    bank.build_from_capsule(cap, env=s["env"], env_params=s["env_params"],
                            backend=s["backend"], params=s["params"],
                            n_frozen=2, prefix_steps=(2,))
    assert len(bank.entries) == 3
    hashes = [e.provenance_hash for e in bank.entries]
    assert len(set(hashes)) == 3
    for entry in bank.entries:
        check = bank.validate_entry(entry, env=s["env"],
                                    env_params=s["env_params"],
                                    backend=s["backend"], params=s["params"])
        assert check["valid"]
    manifest = bank.manifest()
    assert manifest["n_entries"] == 3 and manifest["manifest_hash"]