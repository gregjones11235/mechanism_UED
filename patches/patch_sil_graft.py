p = "/workspace/mechanism_UED/dicode_src/src/dicode/training.py"
src = open(p, newline='').read()
nl = "\r\n" if "\r\n" in src else "\n"
old = (f"    if config.dicode_manager.reset_opt_state:{nl}"
       f"        rl_train_state = _reset_optimizer_state(config, rl_train_state){nl}")
assert src.count(old) == 1, f"anchor x{src.count(old)} (nl={'CRLF' if nl==chr(13)+chr(10) else 'LF'})"
block = nl.join([
    "    # [SIL v1] session-level BC phase on the golden buffer (flag-gated,",
    "    # +training.sil_coef>0 enables; absent/0 = this block is a no-op).",
    '    if float(config.training.get("sil_coef", 0.0) or 0.0) > 0.0:',
    "        from dicode.sil_bc import run_sil_phase",
    "        rl_train_state = run_sil_phase(config, rl_train_state)",
]) + nl
open(p, "w", newline='').write(src.replace(old, old + nl + block))
print(f"graft patch OK (nl={'CRLF' if nl==chr(13)+chr(10) else 'LF'})")
