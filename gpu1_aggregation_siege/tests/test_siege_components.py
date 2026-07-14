#!/usr/bin/env python3
"""S4 SIEGE Component Tests — 8 tests covering all SIEGE modules."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_all():
    from dicode.siege.student_profile import StudentProfileLog
    from dicode.siege.chain_order import ChainOrderLog
    from dicode.siege.focus_quota import FocusQuota
    from dicode.siege.rehearsal import ForgettingRehearsal
    from dicode.siege.siege_notebook import SiegeNotebook
    from dicode.siege.aggregation_integration import chain_completeness_gate

    passed = 0

    # 1: Profile
    p = StudentProfileLog()
    p.update({'collect_wood': 0.96, 'craft_planks': 0.55})
    assert p.get_tier('collect_wood') == 4
    assert not p.is_mastered('craft_planks')
    passed += 1; print(f'  [{passed}] Profile: PASS')

    # 2: Forgetting
    p.update({'collect_wood': 0.80})
    assert p.get_forgetting_risk('collect_wood')
    passed += 1; print(f'  [{passed}] Forgetting: PASS')

    # 3: ChainOrder
    c = ChainOrderLog()
    c.define_chain('test', ['skill_a','skill_b'])
    c.update_from_profile(p)
    assert c.get_break_link('test')['achievement'] == 'skill_a'
    passed += 1; print(f'  [{passed}] ChainOrder: PASS')

    # 4: FocusQuota
    fq = FocusQuota(min_chain_tasks=2)
    check = fq.check(['t1','t2','t3','t4','t5','t6','t7','t8'], ['t1','t9'], 1)
    assert not check['satisfied']
    passed += 1; print(f'  [{passed}] FocusQuota: PASS')

    # 5: Rehearsal
    fr = ForgettingRehearsal()
    assert 'collect_wood' in fr.detect_forgetting(p)
    passed += 1; print(f'  [{passed}] Rehearsal: PASS')

    # 6: SiegeNotebook
    with tempfile.TemporaryDirectory() as td:
        nb = SiegeNotebook(td)
        nb.define_craftax_chains()
        nb.update({'collect_wood': 0.96}, 1000)
        meta = nb.get_candidate_metadata('tx', ['collect_wood'])
        assert meta['siege_wall'] == True
        for k,v in meta.items():
            if isinstance(v,str) and 'tier' in v.lower():
                raise AssertionError(f'Tier label leaked: {k}={v}')
    passed += 1; print(f'  [{passed}] Notebook + no-tier-leak: PASS')

    # 7: Gate
    with tempfile.TemporaryDirectory() as td:
        nb2 = SiegeNotebook(td)
        nb2.define_craftax_chains()
        nb2.update({'collect_wood': 0.96}, 1000)
        meta = {
            'ta': nb2.get_candidate_metadata('ta', ['collect_wood']),
            'tb': nb2.get_candidate_metadata('tb', ['unknown']),
        }
        admitted, rejected, _ = chain_completeness_gate(['ta','tb'], meta, nb2)
        assert 'ta' in admitted and 'tb' in rejected
    passed += 1; print(f'  [{passed}] Gate: PASS')

    # 8: Persistence
    with tempfile.TemporaryDirectory() as td:
        nb3 = SiegeNotebook(td)
        nb3.define_craftax_chains()
        nb3.update({'collect_wood': 0.96}, 1000)
        nb3.save()
        nb4 = SiegeNotebook(td)
        assert nb4.profile.get_tier('collect_wood') == 4
    passed += 1; print(f'  [{passed}] Persistence: PASS')

    print(f'\nALL {passed}/8 TESTS PASSED')
    return 0

if __name__ == '__main__':
    sys.exit(test_all())
