p = "/workspace/mechanism_UED/dicode_src/src/dicode/dreaming/gen_manager.py"
src = open(p, newline='').read()
nl = "\r\n" if "\r\n" in src else "\n"

anchor = "from dicode.dreaming.prompts.dicode.minicraftax_api import context as API_DOCS" + nl
assert src.count(anchor) == 1, f"anchor x{src.count(anchor)}"
assert "_dicode_api_ref" not in src, "already patched"

add = '''
import os as _os_api

# --- auto-generated API reference (flag: DICODE_API_REF=1) -------------------
# minicraftax_api omits Inventory entirely and never lists BlockType /
# Achievement members, so the FM invents names (67 such failures measured).
# Introspected from the real classes, so it cannot drift from source.
def _dicode_api_ref():
    import dataclasses as _dc, importlib as _il
    out = []
    try:
        from craftax.craftax.constants import BlockType as _BT, Achievement as _AC
        out.append("BlockType members -- use exactly these names:\\n"
                   + ", ".join(b.name for b in _BT))
        out.append("Achievement members -- use exactly these names:\\n"
                   + ", ".join(a.name for a in _AC))
    except Exception as _e:
        out.append("(enum introspection failed: %r)" % (_e,))
    specs = (
        ("craftax.craftax.craftax_state", "Inventory",
         "IMPORTANT: pickaxe, sword and armour are integer TIER LEVELS "
         "(1=wood, 2=stone, 3=iron, 4=diamond). There is no iron_pickaxe or "
         "stone_pickaxe field -- set pickaxe=3 for an iron pickaxe."),
        ("minicraftax.craftax_state", "TaskParams", ""),
        ("minicraftax.craftax_state", "EnvState", ""),
    )
    for _mod, _cls, _note in specs:
        try:
            _c = getattr(_il.import_module(_mod), _cls)
            _txt = _cls + " fields (" + _mod + "):\\n" + ", ".join(
                x.name for x in _dc.fields(_c))
            if _note:
                _txt += "\\n" + _note
            out.append(_txt)
        except Exception as _e:
            out.append("(" + _cls + " introspection failed: %r)" % (_e,))
    return "\\n\\n".join(out)

if _os_api.environ.get("DICODE_API_REF", "") == "1":
    _ref = _dicode_api_ref()
    API_DOCS = (API_DOCS
                + "\\n\\n=== EXACT API REFERENCE (auto-generated from source) ===\\n"
                + _ref)
    print("[api-ref] ON: appended %d chars to API_DOCS" % len(_ref))
# ----------------------------------------------------------------------------
'''.replace("\n", nl)

open(p, "w", newline='').write(src.replace(anchor, anchor + add))
print("api-ref patch OK")
