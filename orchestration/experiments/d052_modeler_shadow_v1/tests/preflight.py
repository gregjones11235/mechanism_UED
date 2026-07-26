import sys, time
sys.path.insert(0, sys.argv[1])
import llm_client as L

probes = [
    ("ds", "deepseek-v4-pro"),
    ("qw", "qwen-flash-2025-07-28"),
    ("gl", "glm-4-flash"),
]
for prov, model in probes:
    t0 = time.time()
    r = L.api(prov, model, [{"role": "user", "content": "Reply with the single word: ok"}], mtok=8)
    dt = time.time() - t0
    if r.get("ok"):
        print("PROBE %-3s %-22s OK   %.1fs  content=%r  mrt=%s" % (prov, model, dt, (r.get("content") or "")[:40], r.get("mrt")))
    else:
        print("PROBE %-3s %-22s FAIL %.1fs  err=%s" % (prov, model, dt, r.get("err")))
