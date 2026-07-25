p = "/workspace/mechanism_UED/dicode_src/experiments/training/run_dicode.py"
src = open(p, newline='').read()
nl = "\r\n" if "\r\n" in src else "\n"
anchor = '                    prereq_threshold=_sp.get("prereq_threshold", 0.3),' + nl
assert src.count(anchor) == 1, f"anchor x{src.count(anchor)}"
assert "max_target_achievements=int(_sp" not in src, "already patched"
add = '                    max_target_achievements=int(_sp.get("max_target_achievements", 6)),' + nl
open(p, "w", newline='').write(src.replace(anchor, anchor + add))
print("max-targets patch OK")
