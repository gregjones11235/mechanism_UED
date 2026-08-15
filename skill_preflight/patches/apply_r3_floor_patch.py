#!/usr/bin/env python3
"""Surgical arm patch: add +skill_preflight.r3_floor_clause (default true = byte-identical).

When false, a starting floor is NOT counted as scaffolding-away a prerequisite in R3.
Premark and inventory channels of R3 stay live; R1/R2 untouched.

Edits exactly two files with unique-anchor replacements:
  src/dicode/skill_preflight/scaffold_gate.py   (4 anchors)
  src/dicode/evolution_efficient.py             (4 anchors)

Run from dicode_src root:  python3 apply_r3_floor_patch.py
Refuses to run twice (idempotence guard). Backs up both files to *.pre_r3floor.
Finishes with a behavior battery: pristine vs patched on default flag must be identical,
and the four surgical cases must land exactly as pre-registered.
"""
import hashlib, importlib.util, os, shutil, sys

ROOT = os.getcwd()
GATE = os.path.join(ROOT, "src/dicode/skill_preflight/scaffold_gate.py")
EVOL = os.path.join(ROOT, "src/dicode/evolution_efficient.py")

EDITS = {
    GATE: [
        # 1) signature: new kwarg after the exemption kwarg
        (
            "    mastered_prereq_exemption: bool = False,\n) -> GateVerdict:",
            "    mastered_prereq_exemption: bool = False,\n"
            "    r3_floor_clause: bool = True,\n"
            ") -> GateVerdict:",
        ),
        # 2) docstring: document the new kwarg right after the exemption paragraph
        (
            "            iron/gnomish consolidation deficit to. Default False -> v1 byte-identical.\n",
            "            iron/gnomish consolidation deficit to. Default False -> v1 byte-identical.\n"
            "        r3_floor_clause: [surgical ablation, 2026-08-10] when False, a starting floor\n"
            "            is NOT counted as scaffolding-away a prerequisite in R3 (the premark and\n"
            "            inventory channels of R3, and R1/R2, are unchanged). Motivated by the\n"
            "            matched-seed archive audit: the gate's LLM repair rewrote 24/27 trained\n"
            "            mob levels from floor 2 to floor 0, starving the floor-2 cluster.\n"
            "            Default True -> byte-identical to the shipped gate.\n",
        ),
        # 3) the cut itself
        (
            "    scaffolded = premarked | inv_grants | flr_grants\n",
            "    scaffolded = premarked | inv_grants | (flr_grants if r3_floor_clause else frozenset())\n",
        ),
        # 4) evidence: never indict the floor when the clause is off (else the repair
        #    prompt would still instruct the LLM to strip the floor on premark/inventory hits)
        (
            "                if h in flr_grants:\n",
            "                if r3_floor_clause and h in flr_grants:\n",
        ),
    ],
    EVOL: [
        # 5) read the flag next to its sibling
        (
            '            _r3_exempt = bool(_sp_cfg.get("r3_mastered_exemption", False))\n',
            '            _r3_exempt = bool(_sp_cfg.get("r3_mastered_exemption", False))\n'
            '            _r3_floor = bool(_sp_cfg.get("r3_floor_clause", True))\n',
        ),
        # 6) first check_code call
        (
            "                verdict = check_code(code, _snapshot, mastered_prereq_exemption=_r3_exempt)\n",
            "                verdict = check_code(code, _snapshot, mastered_prereq_exemption=_r3_exempt, r3_floor_clause=_r3_floor)\n",
        ),
        # 7) re-check call
        (
            "                    re_verdict = check_code(new_code, _snapshot, mastered_prereq_exemption=_r3_exempt)\n",
            "                    re_verdict = check_code(new_code, _snapshot, mastered_prereq_exemption=_r3_exempt, r3_floor_clause=_r3_floor)\n",
        ),
        # 8) per-cycle provenance suffix (default arm prints byte-identical line)
        (
            '            print(\n'
            '                f"    WORKER: [ScaffoldGate] checked {_n_checked}, violations "\n'
            '                f"{_n_violated}, repaired {_n_repaired}, dropped {_n_dropped}."\n'
            '            )\n',
            '            _suffix = "" if _r3_floor else "  [R3-floor OFF]"\n'
            '            print(\n'
            '                f"    WORKER: [ScaffoldGate] checked {_n_checked}, violations "\n'
            '                f"{_n_violated}, repaired {_n_repaired}, dropped {_n_dropped}.{_suffix}"\n'
            '            )\n',
        ),
    ],
}

# ---------------------------------------------------------------- apply
for path, edits in EDITS.items():
    src = open(path).read()
    if "r3_floor_clause" in src:
        sys.exit(f"REFUSED: {path} already contains r3_floor_clause (patch applied before?)")
    for old, _ in edits:
        n = src.count(old)
        assert n == 1, f"ANCHOR NOT UNIQUE ({n} hits) in {path}:\n{old!r}"

pristine_gate = open(GATE).read()          # keep for the battery
for path, edits in EDITS.items():
    src = open(path).read()
    shutil.copy(path, path + ".pre_r3floor")
    for old, new in edits:
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched {os.path.relpath(path)}   md5 {hashlib.md5(src.encode()).hexdigest()[:12]}")

# ---------------------------------------------------------------- battery
def load(name, source):
    tmp = os.path.join(ROOT, f"_{name}_battery.py")
    open(tmp, "w").write(source)
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass decorators resolve cls.__module__ here
    try:
        spec.loader.exec_module(mod)
    finally:
        os.remove(tmp)
    return mod

sys.path.insert(0, os.path.join(ROOT, "src"))
def _restore_and_die(msg):
    print("\nBATTERY ABORTED (" + msg + ") — restoring backups:")
    for p in EDITS: shutil.copy(p + ".pre_r3floor", p)
    sys.exit(1)
try:
    old_gate = load("gate_pristine", pristine_gate)
    new_gate = load("gate_patched", open(GATE).read())
except Exception as e:
    _restore_and_die(f"import failure: {e!r}")

SR = {"defeat_gnome_warrior": 0.0, "enter_gnomish_mines": 0.20, "enter_dungeon": 0.90,
      "collect_iron": 0.30, "make_stone_pickaxe": 0.50, "collect_wood": 0.95}
def lvl(relevant, premark=(), floor=0, inv=""):
    return (
        "class T:\n"
        "    def build(self, builder):\n"
        f"        self.relevant_achievements = [{', '.join('Achievement.'+r for r in relevant)}]\n"
        + (f"        self.completed_achievements = [{', '.join('Achievement.'+p for p in premark)}]\n" if premark else "")
        + (f"        builder.set_player_inventory({inv})\n" if inv else "")
        + (f"        builder.set_starting_floor({floor})\n" if floor else "")
    )

C_FLOOR = lvl(["DEFEAT_GNOME_WARRIOR"], floor=2)                       # the pathology
C_PREMK = lvl(["DEFEAT_GNOME_WARRIOR"], premark=["ENTER_GNOMISH_MINES"])
C_INV   = lvl(["COLLECT_IRON"], inv='{"pickaxe": 2}')
C_R1R2  = lvl(["DEFEAT_GNOME_WARRIOR"], premark=["COLLECT_WOOD", "DEFEAT_GNOME_WARRIOR"])
battery = [("floor-mob", C_FLOOR), ("premark-prereq", C_PREMK),
           ("inventory-prereq", C_INV), ("r1r2", C_R1R2)]

fails = []
# A) default flag: pristine and patched must agree verdict-for-verdict
for name, code in battery:
    a = old_gate.check_code(code, SR)
    b = new_gate.check_code(code, SR)
    if (a.ok, a.violations, a.evidence) != (b.ok, b.violations, b.evidence):
        fails.append(f"DEFAULT-PATH DIVERGENCE on {name}: {a.violations} vs {b.violations}")
# B) surgical flag: pre-registered expectations
s_floor = new_gate.check_code(C_FLOOR, SR, r3_floor_clause=False)
if not s_floor.ok:
    fails.append(f"surgical: floor-mob should PASS, got {s_floor.violations}")
s_premk = new_gate.check_code(C_PREMK, SR, r3_floor_clause=False)
if "R3_focus_prereq_scaffolded" not in s_premk.violations:
    fails.append("surgical: premark channel of R3 must still fire")
s_inv = new_gate.check_code(C_INV, SR, r3_floor_clause=False)
if "R3_focus_prereq_scaffolded" not in s_inv.violations:
    fails.append("surgical: inventory channel of R3 must still fire")
s_r12 = new_gate.check_code(C_R1R2, SR, r3_floor_clause=False)
if not {"R1_premark_mastered", "R2_focus_premarked"} <= set(s_r12.violations):
    fails.append(f"surgical: R1/R2 must be untouched, got {s_r12.violations}")
# C) evidence hygiene: with the clause off, no floor indictment anywhere
mixed = lvl(["DEFEAT_GNOME_WARRIOR"], premark=["ENTER_GNOMISH_MINES"], floor=2)
s_mix = new_gate.check_code(mixed, SR, r3_floor_clause=False)
if "set_starting_floor" in s_mix.evidence:
    fails.append("surgical: evidence must not indict the floor when the clause is off")

if fails:
    _restore_and_die("; ".join(fails))
print("\nBATTERY OK: default path identical on all 4 cases; surgical semantics as pre-registered.")
print("backups at *.pre_r3floor  |  launch override: +skill_preflight.r3_floor_clause=false")
