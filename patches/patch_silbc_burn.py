p = "/workspace/mechanism_UED/dicode_src/src/dicode/sil_bc.py"
src = open(p, newline='').read()
old = ('    burn = int(tcfg.window_mem)\n'
       '    assert burn < K, f"burn-in {burn} >= segment length {K}"\n')
new = ('    burn = tcfg.get("sil_burn", None)\n'
       '    if burn is None:\n'
       '        # window_mem (=128) exceeds segment length K (=64): burn all pre-\n'
       '        # crossing steps, loss covers the post-crossing tail. Card S5\n'
       '        # amended 7/21; +training.sil_burn overrides (0 = ablation needle).\n'
       '        burn = min(int(tcfg.window_mem), K - 16)\n'
       '    burn = int(burn)\n'
       '    assert 0 <= burn < K, f"burn-in {burn} vs segment length {K}"\n')
assert src.count(old) == 1, f"anchor x{src.count(old)}"
open(p, "w", newline='').write(src.replace(old, new))
print("burn patch OK")
