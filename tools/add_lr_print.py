p = "/workspace/mechanism_UED/dicode_src/src/dicode/utils/general/train_state_utils.py"
src = open(p, newline='').read()
if "LR at restored count" in src:
    print("already present, skip")
else:
    nl = "\r\n" if "\r\n" in src else "\n"
    anchor = 'print(f"  > VERIFICATION: Restored Optimizer Step Count = {step_count}")'
    assert src.count(anchor) == 1, f"anchor count = {src.count(anchor)}"
    indent = src.split(anchor)[0].rsplit(nl,1)[1]
    addition = anchor + nl + indent + 'print(f"  > VERIFICATION: LR at restored count = {float(linear_schedule(int(step_count))):.3e}")'
    open(p, "w", newline='').write(src.replace(anchor, addition))
    print("LR print added")
