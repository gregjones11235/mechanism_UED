p = "/workspace/mechanism_UED/dicode_src/src/dicode/dreaming/gen_manager.py"
src = open(p, newline='').read()
nl = "\r\n" if "\r\n" in src else "\n"

anchor = "from dicode.dreaming.prompts.dicode.world_gen_nl import context as WORLD_GEN" + nl
assert src.count(anchor) == 1, f"anchor x{src.count(anchor)}"
assert "_dicode_floor_guide" not in src, "already patched"

add = '''
import os as _os

# --- floor-guide reveal (experiment flag: DICODE_TUTORIAL_REVEAL=1) ----------
# Appends the tutorial paragraphs for floors the agent has ALREADY demonstrably
# reached to the natural-language mechanics block. Floors it has never reached
# are withheld, so no depth structure is revealed that the student has not
# itself discovered. Off unless the env var is set.
_DICODE_GUIDE_SECTIONS = ("Basic Mechanics", "Floor 0", "Floor 1", "Floor 2")

def _dicode_floor_guide(sections=_DICODE_GUIDE_SECTIONS):
    import re as _re
    from dicode.dreaming.prompts.cl_.knowledge_base_designer import (
        knowledge_base_designer as _kb_raw,
    )
    _kb = _kb_raw.replace("\\r\\n", "\\n")
    _out = []
    for _m in _re.finditer(r"(?m)^## (.+)$", _kb):
        _name = _m.group(1).strip()
        if not any(_name.startswith(_s) for _s in sections):
            continue
        _nxt = _kb.find("\\n## ", _m.end())
        _out.append("## " + _name + _kb[_m.end(): _nxt if _nxt > 0 else len(_kb)])
    return "\\n".join(_out)

if _os.environ.get("DICODE_TUTORIAL_REVEAL", "") == "1":
    _guide = _dicode_floor_guide()
    GAME_MECHANICS = (
        GAME_MECHANICS
        + "\\n\\n=== FLOOR GUIDE (floors the agent has demonstrably reached) ===\\n"
        + _guide
    )
    print(f"[tutorial-reveal] ON: appended {len(_guide)} chars to GAME_MECHANICS")
# ----------------------------------------------------------------------------
'''.replace("\n", nl)

src = src.replace(anchor, anchor + add)
open(p, "w", newline='').write(src)
print("tutorial-reveal patch OK")
