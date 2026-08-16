import pytest
import numpy as np
from types import SimpleNamespace


def test_train_key_exact_components_and_fail_closed():
    mod = pytest.importorskip("dicode.ppo_tr")
    cfg = {"x": 1}
    state = {"p": np.zeros((2,), dtype=np.float32)}
    rng = np.zeros((2,), dtype=np.uint32)
    sig = (("a", "codehash"), ("original_craftax", "orig"))
    key = mod.train_compile_cache_key(cfg, sig, 2, np.zeros((1, 2), np.float32), np.ones((2,)), state, rng)
    assert key is not None
    assert mod.train_compile_cache_key(cfg, None, 2, None, None, state, rng) is None
    assert key != mod.train_compile_cache_key(cfg, tuple(reversed(sig)), 2, np.zeros((1, 2), np.float32), np.ones((2,)), state, rng)


def test_train_cache_config_default_off():
    from pathlib import Path
    text = Path(__file__).parents[4].joinpath("conf", "config.yaml").read_text()
    assert "train_compile_cache: false" in text

def test_compile_helper_once_disabled_and_lru():
    mod = pytest.importorskip("dicode.ppo_tr")
    mod.clear_train_compile_cache(); calls=[]
    class J:
        def lower(self,*a): calls.append(a); return self
        def compile(self): return self
    builds=[]
    def build(): builds.append(1); return J()
    mod._get_or_compile_train(None, build, (1,), False, 2); mod._get_or_compile_train(None, build, (1,), False, 2)
    assert len(builds)==2 and calls==[]
    mod.clear_train_compile_cache(); builds.clear(); calls.clear()
    _,h1=mod._get_or_compile_train("a", build, (1,), True, 1); _,h2=mod._get_or_compile_train("a", build, (1,), True, 1)
    assert len(builds)==1 and len(calls)==1 and (h1,h2)==(False,True)
    mod.clear_train_compile_cache(); builds.clear(); calls.clear()
    mod._get_or_compile_train("a", build, (1,), True, 1); mod._get_or_compile_train("b", build, (1,), True, 1); mod._get_or_compile_train("a", build, (1,), True, 1)
    assert len(builds)==3 and len(calls)==3

def test_key_components_and_global_step_value_ignored():
    mod = pytest.importorskip("dicode.ppo_tr")
    base = dict(config={"x":1}, task_signature=(("a","h"),), num_training_updates=2,
        task_embeddings=np.zeros((1,2),np.float32), task_distribution_proportions=np.ones(2),
        train_state={"p":np.zeros(2,np.float32)}, rng=np.zeros(2,np.uint32), current_original_return=np.array(1.,np.float32), global_update_step=np.array(3,np.int32))
    k=mod.train_compile_cache_key(**base)
    for field, val in [("config",{"x":2}),("task_signature",(("b","h"),)),("num_training_updates",3),
        ("task_embeddings",np.ones((1,2),np.float32)),("task_distribution_proportions",np.zeros(2)),
        ("train_state",{"p":np.zeros(3,np.float32)}),("rng",np.zeros(3,np.uint32))]:
        changed=dict(base); changed[field]=val; assert k != mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["global_update_step"]=np.array(99,np.int32); assert k == mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["current_original_return"]=np.array(99.,np.float32); assert k == mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["global_update_step"]=np.array([3],np.int32); assert k != mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["current_original_return"]=np.array([1.],np.float32); assert k != mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["train_state"]={"p":np.ones(2,np.float32)}; assert k == mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["rng"]=np.ones(2,np.uint32); assert k == mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["global_update_step"]=99; assert k != mod.train_compile_cache_key(**changed)
    changed=dict(base); changed["task_signature"]=(("a","otherhash"),); assert k != mod.train_compile_cache_key(**changed)
    if hasattr(mod, "jax"):
        changed=dict(base); changed["current_original_return"]=1.0; assert k != mod.train_compile_cache_key(**changed)

def test_task_signature_builder_on_off_and_order():
    mod=pytest.importorskip("dicode.training")
    class A:
        def __init__(self): self.calls=0
        def get_task_codes(self, ids): self.calls+=1; return {i:i for i in ids}
    a=A(); cfg={"performance":{"train_compile_cache":False}}
    assert mod._maybe_build_task_signature(cfg,a,["a"],object) is None and a.calls==0
    class Original: pass
    cfg["performance"]["train_compile_cache"]=True; sig=mod._maybe_build_task_signature(cfg,a,["a","b"],Original); assert a.calls==1 and sig[0][0]=="a" and sig[-1][0]=="original_craftax"
    b=A(); sig2=mod._maybe_build_task_signature(cfg,b,["b","a"],Original); assert b.calls==1 and sig != sig2

def test_resolve_global_step_no_input_mutation():
    mod=pytest.importorskip("dicode.ppo_tr")
    assert mod._resolve_global_step(5,None,2)==7 and mod._resolve_global_step(5,9,2)==11
    if hasattr(mod, "jax"):
        assert int(mod.jax.jit(lambda d,u: mod._resolve_global_step(5,d,u))(mod.jnp.array(9),mod.jnp.array(2))) == 11

def test_default_flag_and_profiling_cache_path_source():
    from pathlib import Path
    text=Path(__file__).parents[4].joinpath("conf","config.yaml").read_text(); assert "train_compile_cache: false" in text
    src=Path(__file__).parents[2].joinpath("ppo_tr.py").read_text(); assert "if profiling and not cache_on" in src

def test_signature_missing_source_fails_closed():
    mod=pytest.importorskip("dicode.training")
    class A:
        def get_task_codes(self, ids): return {}
    assert mod._maybe_build_task_signature({"performance":{"train_compile_cache":True}}, A(), ["missing"], object) is None

def test_run_training_session_mode_matrix(monkeypatch):
    mod=pytest.importorskip("dicode.ppo_tr")
    class C:
        training=SimpleNamespace(x=1)
        def __init__(self, cache, profile): self.data={"performance":{"train_compile_cache":cache},"runtime_profiling":{"enabled":profile}}
        def get(self,k,d=None): return self.data.get(k,d)
    counts={"build":0,"lower":0,"args":[]}
    class Comp:
        def lower(self,*a): counts["lower"]+=1; return self
        def compile(self): return self
        def __call__(self,*a): counts["args"].append(len(a)); return {"train_state":None,"metrics":{"num_updates_done":0,"num_env_steps_done":0}}
    monkeypatch.setattr(mod,"make_train",lambda *a: (lambda *x: None)); monkeypatch.setattr(mod.jax,"jit",lambda f: (counts.__setitem__("build",counts["build"]+1) or Comp()))
    for cache,profile in ((False,False),(False,True),(True,False),(True,True)):
        mod.clear_train_compile_cache(); before=dict(counts); a0=len(counts["args"])
        cfg=C(cache,profile); mod.run_training_session(cfg, np.zeros(2,np.uint32), [], 1, train_state=None, task_signature=(("a","h"),)); mod.run_training_session(cfg, np.zeros(2,np.uint32), [], 1, train_state=None, task_signature=(("a","h"),))
        expected={(False,False):(2,0,[3,3]),(False,True):(2,2,[3,3]),(True,False):(1,1,[4,4]),(True,True):(1,1,[4,4])}[(cache,profile)]
        assert counts["build"]-before["build"] == expected[0]
        assert counts["lower"]-before["lower"] == expected[1]
        assert counts["args"][a0:] == expected[2]
