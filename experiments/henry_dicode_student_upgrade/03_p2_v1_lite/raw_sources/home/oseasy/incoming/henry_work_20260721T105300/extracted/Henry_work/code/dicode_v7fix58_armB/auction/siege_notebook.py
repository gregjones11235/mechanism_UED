"""SiegeNotebook — the modeler's PERSISTENT siege journal (v6, v6_design.md §3.5).

v6's central data structure. Where v5's StudentProfileLog is a passive time series the modeler
re-reads from scratch every session, the SiegeNotebook is a *growing* journal the modeler maintains
ACROSS sessions — like a human player's mental notes when stuck on a boss: which fight I'm currently
attacking, which prerequisites in its chain I've already nailed vs still can't do, which chains I've
already conquered, and which skills I must keep rehearsing so I don't forget them.

WHY PERSISTENT (v6_design.md §3.5, not re-derived each session):
  1. (c) co-occurrence evidence is sparse (tier3 SR ~10% -> few successful episodes per session) —
     must accumulate across sessions to be stable.
  2. conquering one hard fight (with its backtracked sub-goals) spans MANY sessions — must remember
     "which fight I'm attacking / which link of the chain / which links are already consolidated",
     else the curriculum thrashes back into v5's spread-thin disease.
  3. H1's "self-style" accumulation is itself memory-dependent.

A+B HYBRID (v6_design.md §3.5, user 2026-07-04):
  - B (CODE guarantees, anti-drift backstop): schema + cross-session persistence, the §3.2 SCOPE
    hard constraint (focus may NOT be an easy/saturated skill), per-link "consolidated/unmastered"
    flags computed by an SR THRESHOLD (NOT trusted from the LLM's self-report), and the minimum
    condition for switching focus (K consecutive non-improving sessions). These CANNOT be violated
    by the LLM.
  - A (LLM judgement, "player intuition"): reads the previous notebook + this session's new evidence
    and PROPOSES an updated notebook (schema-constrained). It does not rewrite from zero.

This module owns B. The LLM (A) produces a raw proposed-update dict; ``apply_llm_update`` runs it
through every B-layer rule before it is allowed to persist. So a hallucinated "focus = collect_wood"
or a "this link is mastered" claim that the SR data contradicts is silently corrected here, not
trusted.

Storage: a plain JSON file ``siege_notebook.json`` in the run's cwd (same place as
``student_profile_history.json`` / ``task_graph.graphml``), atomic-written so an interrupted write
can't corrupt it, and reloaded on resume.

NOTE ON SR SOURCE (v6_design.md §3.8): mastery flags this phase are computed from the held-out
AGGREGATE per-skill SR (from StudentProfileLog). The (c) per-episode co-occurrence signal does not
exist yet (evaluation throws away per-env multi-hot at .sum()); wiring it in is a later phase. The
mastery-flag computation is deliberately funnelled through ``_mastery_from_sr`` so that later phase
can enrich it without touching callers.
"""

from __future__ import annotations

import json
import os
import re

_DEFAULT_PATH = "siege_notebook.json"

# --- B-layer thresholds (the "code guarantees") --------------------------------------------------
# These module constants are the DEFAULTS. Every one is now overridable per-run via config
# (dicode_manager.siege_* keys) — see SiegeNotebook.__init__ / SiegeThresholds and gen_manager, which
# passes the config in. The constants stay as the fallback so tests and any code that constructs a
# SiegeNotebook() with no thresholds keep the documented default behaviour byte-for-byte.
# ★ These are the highest-leverage knobs (user 2026-07-05): MASTERED_SR gates link compression,
#   and MATURITY_* gates whether a siege ever opens.

# A prereq LINK is CONSOLIDATED/MASTERED (safe to compress into Completed, per §3.4) only when its
# held-out SR is at/above this. Mid-chain links below this MUST stay Relevant (trained), never
# compressed — that was exactly the v5 chain-severing disease.
# NOTE: this judges a LINK ("has this prerequisite been learned"); it is INDEPENDENT of SATURATED_SR,
# which judges a FOCUS candidate ("is this skill too easy to be a wall"). They are different questions
# on different objects and need not share a value (user 2026-07-05).
MASTERED_SR = 70.0

# A skill is UNMASTERED (a live siege link that must be trained in-level) when its SR is at/below this.
# Between UNMASTERED_SR and MASTERED_SR is the "rising / in-progress" band.
UNMASTERED_SR = 20.0

# §3.2 SCOPE: the focus must be a genuine hard wall, NEVER an easy/saturated skill. A skill at/above
# this SR is "already-held ground", forbidden as a focus.
# ★ user 2026-07-05 (2nd round): combat has NO 40% ceiling (a human expert beats a hard boss near
#   100%), so "saturated / already ground" is a HIGH bar for BOTH combat and gear — set to 80, NOT
#   bound to MASTERED_SR. A combat skill at 40% is still a wall worth attacking, not saturated.
SATURATED_SR = 80.0

# --- EARLY-TRAINING guard (user 2026-07-04): "low SR" has TWO causes and only ONE is a real wall ---
# (A) genuine wall: student PAST the early phase, prerequisites in place, STILL can't do it (siege it).
# (B) early-immature: whole student still weak, this skill just hasn't had its turn (NORMAL_EARLY —
#     never siege it). Until the student crosses a maturity threshold, NO focus may be set (a low-SR
#     skill early is immaturity, not a wall). Judged purely from SR statistics — no tier table.
MATURITY_MIN_SNAPSHOTS = 4     # fewer snapshots than this = too early to call anything a wall
# ★ user 2026-07-05: relaxed so the siege can OPEN in the real tier3~low-ceiling regime (the old
#   12×50 could mean the siege never opens). A "decent" skill now counts at 35, and 10 of them suffice.
# ★ user 2026-07-05, from REAL v5 curve (dicode-v5yA-s0-r2, job 3640867): tier1 has 15 skills at
#   SR>=50 by session ~9 and tier2 climbs to ~10 by session ~17 (the burst->plateau knee, return ~30).
#   20 = 15(tier1) + ~5(tier2) at SR>=50 -> siege opens right at that knee, once the base is laid.
MATURITY_MIN_MASTERED = 20     # fewer skills at/above MATURITY_SKILL_SR = still in the early ramp
MATURITY_SKILL_SR = 50.0       # a skill counts toward maturity once it is at/above this SR

# (v7fix3.1 audit: the fix4-era FOCUS_MIN_STALL_SESSIONS knob and its per-focus stall_sessions
# ratchet were removed — written everywhere, read nowhere; ladder escalation and retirement are
# driven entirely by the FROZEN counter below.)

# How much the focus's SR must rise between two recorded sessions to count as "improving" (resets the
# stall counter). Small, to tolerate eval noise but still register real progress.
FOCUS_IMPROVE_PP = 3.0

# --- v6fix7 P1a: ESCALATION LADDER (adaptive patience; user 2026-07-06) ----------------------------
# The old binary retire/reopen loop watched ONLY the focus's own SR: a tier4 wall whose SR sits at 0
# while its foundations climb was mis-read as "stalled", and fix4's iron-pickaxe siege showed the
# retire->same-session-reopen cycle. The ladder counts FROZEN sessions instead — a session is frozen
# only when the WHOLE attack tree shows no progress (focus SR flat AND every link flat AND no
# chain-frontier advance from the failure-episode signal). Patience is therefore unbounded while
# anything measurable is moving; the ladder escalates the response only on total freeze:
#   L1 (>= LADDER_L1 frozen): asked to switch form OR defend staying with a NEW plan (accountability);
#   L2 (>= LADDER_L2): form switch FORCED (DEPTH<->CONSOLIDATE) — code overrides;
#   L3 (>= LADDER_L3): tactic revision FORCED (style_note must be materially different);
#   L4 (>= LADDER_L4): retire + cooldown; repeated retirements blacklist until NEW chain evidence.
LADDER_L1 = 3
LADDER_L2 = 6
LADDER_L3 = 9
LADDER_L4 = 12
# Windowed-slope progress signal (defeats the fix4 best-SR ratchet: oscillation under an old best is
# NOT progress; a genuine climb shows a positive recent slope even without a new best).
SLOPE_WINDOW = 6           # readings in the regression window
SLOPE_MIN_PP = 1.0         # pp/session; slope above this counts as live progress
# Retirement aftermath.
COOLDOWN_SESSIONS = 6      # a retired wall may not reopen for this many sessions
BLACKLIST_RETIREMENTS = 2  # retired this many times -> blacklisted until new chain evidence
# Conquest (#8 fix): verified_chains "verified" status + protected_set admission require the wall to
# HOLD at mastered_sr for this many consecutive snapshots — a one-session +delta record is progress,
# never conquest (fix4 wrote make_iron_pickaxe into verified_chains/protected at 44%).
CONQUEST_CONSECUTIVE = 2
# v6fix7 P1c (AutoManual-lite): a style_note that goes this many sessions without the modeler's
# evidence_check reporting "supported" is marked STALE — the journal then demands re-derivation
# instead of further refinement of an unverified guess (self-reinforcing-tactic loop breaker).
NOTE_STALE_SESSIONS = 4

# --- §2.6 MULTI-FOCUS (user 2026-07-05, 3rd round): parallel sieges + cross-achievement assist ------
# The siege may attack up to MAX_FOCUS walls at once. A NEW focus may be opened only once ANY existing
# focus has reached FOCUS_EXPAND_SR — "one is going well, so there's slack to also carry a new one".
# The new focus may be ANY tier3+ achievement (incl. gear: upgrading gear helps later attacks).
MAX_FOCUS = 3
FOCUS_EXPAND_SR = 50.0

# --- v6fix8 ①: FAST-CLIMB GRADUATION (focus economics; user 2026-07-07) ----------------------------
# fix7 post-mortem: make_iron_pickaxe climbed 0->54% held-out in ~6 siege decisions yet kept ALL its
# siege privileges (guaranteed generation seats, CAS exemption, training quota) for 17 sessions,
# because the only exits were conquest (>=70 twice — unreachable in the 44-61 oscillation dead zone)
# and total-freeze retirement (never fires while the wall is moving). Meanwhile the baseline learned
# the same skill NATURALLY (62.9% @step1933, faster than the siege). A wall that climbs fast was
# never a real wall — the siege's job (get it over the hump) is done, and holding the monopoly past
# that point only taxes the rest of the curriculum.
# GRADUATION: a focus whose held-out SR holds >= GRADUATE_SR for GRADUATE_CONSECUTIVE consecutive
# snapshots moves to MAINTENANCE: it leaves ``foci`` (all siege privileges withdraw automatically),
# its wall+links join the protected_set (forgetting-triggered rehearsal holds the ground; the
# verified_chains entry stays status='progress' — graduation is NOT conquest, #8 semantics kept).
# Re-siege is legal only after a REAL collapse (SR < GRADUATE_SR - MAINT_RESIEGE_DROP_PP).
GRADUATE_SR = 50.0            # held-out SR at/above this ...
GRADUATE_CONSECUTIVE = 2      # ... for this many consecutive snapshots -> maintenance
MAINT_RESIEGE_DROP_PP = 15.0  # a maintained wall may be re-sieged only below graduate_sr - this

# --- v6fix8 ③: DRILL-TRANSFER GAP hard gate (P1b.3 completed; user 2026-07-07) ---------------------
# fix7's gap hint was prompt-only: the modeler SAW "trained 100% vs held-out 48% = 52pp" (it wrote
# exactly that into its style_note) and still kept refining the same calm-sandbox drills. The gate:
# when a focus's best siege-level TRAINED SR >= GAP_TRAINED_MIN while the wall's held-out SR lags by
# >= GAP_MIN_PP for GAP_FORCE_SESSIONS consecutive siege decisions, the attack form for that wall is
# FORCED to DEPTH (full-pressure levels; no more isolation drills) via the same code-enforced
# required_form channel the ladder L2 uses. Cleared as soon as the gap closes or the drill regresses.
GAP_TRAINED_MIN = 90.0        # the drill is "won in its sandbox" at/above this trained SR
GAP_MIN_PP = 30.0             # trained-vs-held-out gap at/above this = not transferring
GAP_FORCE_SESSIONS = 3        # consecutive over-gap siege decisions before the form is forced

# v6fix9 P3 — gap EARLY-STOP ("fixable overfit" vs "style-driven refusal"). fix8 ③ could convict a
# wall of sandbox overfit and force DEPTH, but never asked whether the forcing WORKED: armour sat in
# forced-DEPTH for 8 decisions with held-out flat (10-21, all noise) until the enabler budget
# finally caught it — ~5 wasted decisions of siege firepower. Entering gap_forced snapshots the
# held-out SR as a baseline; each further over-gap decision where held-out has NOT risen by
# GAP_STALL_MIN_GAIN_PP (same +3pp noise floor as the P1a ratchet: held-out n=1024 -> se ~1.5pp)
# counts a stall, and any real movement re-baselines AND resets the count (the ratchet subsumes a
# separate slope test). At GAP_STALL_PATIENCE stalls (mirrors GAP_FORCE_SESSIONS: 3 readings to
# convict overfit, 3 forced decisions to convict "unfixable") the wall is STYLE-REJECTED and takes
# the NORMAL retirement path (_archive_retirement: cooldown + blacklist + failed-tactic archive).
# COMBAT walls are NOT exempt: gap_forced presupposes trained >= 90% (the sandbox already works),
# which is never true of a tier4 wall grinding at 0% — the long-siege patience case cannot get here.
GAP_STALL_PATIENCE = 3        # forced-DEPTH decisions with no held-out movement before retirement
GAP_STALL_MIN_GAIN_PP = 3.0   # held-out must rise this far over the forced-entry baseline to count

# --- v6fix8 ⑤: ENABLER SIEGE BUDGET (class-aware patience; user 2026-07-07) ------------------------
# v6's H1 lives in COMBAT walls; gear/craft walls are enablers. fix7's modeler nevertheless spent its
# ONLY focus slot on an enabler for 17 sessions. COMBAT walls keep the unbounded-patience ladder
# (a tier4 fight may sit at 0% for 20+ sessions while foundations rise — that is a healthy long
# siege). A NON-combat (enabler) focus instead carries a bounded budget: after ENABLER_MAX_SESSIONS
# siege decisions it is retired through the NORMAL retirement machinery (cooldown + blacklist +
# new-tactic-on-reopen all apply) — the natural curriculum finishes cheap walls; siege slots are for
# walls that cannot fall any other way. Classification is family_of() (a category label, not a
# course-chain prior — same boundary as the combat_targets list the modeler already receives).
ENABLER_MAX_SESSIONS = 8      # siege decisions (one per siege session) an enabler focus may consume

# --- v6fix10: reachability economics (door-first / yield / discount / high-water) ------------------
# Constitution (one line): the siege's legitimate scope is lifting a skill OUT of the p~0 dead zone
# into the learnability band; once a skill is in the band and moving, the normal curriculum owns it.
# (Unifies the fix7 iron-pickaxe fake wall, the fix9 kobold door, and the fix8 gnome positive case.)
# ① DOOR GATE: a wall whose failures' TOP missing link sits below DOOR_MIN_SR held-out is not
#   attackable "through the door" — the door ITSELF becomes the focus (gateway_for=<wall>).
#   v6fix10.1: the scan covers the top-3 missing links (a closed door at rank 2/3 behind an open
#   rank-1 link is the same disease); ranks past the first need DOOR_MIN_SHARE of failures to count
#   (noise guard). A candidate with NO failure forensics yet is NOT opened blind — it is parked in
#   ``pending_track`` for one session of chain mining first (the deployed fix10 silently waived the
#   gate exactly for first-open walls, the kobold-at-s9 shape the gate was built for).
DOOR_MIN_SR = 10.0
DOOR_MIN_SHARE = 0.15
PENDING_TRACK_TTL = 8  # sessions a pending-track candidate stays mined without being re-proposed
# v6fix10.1 hazard-5: a gateway's wall leaves the auto-open blocklist once the door is INSIDE the
#   learnable band (>= GATEWAY_RELEASE_SR) — requiring full door graduation (>=50x2) could lock a
#   gnome-type wall out for the whole run when the door's natural plateau sits mid-band (mines:
#   baseline s151 = 21.7%). fix8 attacked gnome at mines ~6% and produced the only H1 positive;
#   past 20% the wall may compete again (③ discount + ④ + the armed P3 police a premature reopen).
GATEWAY_RELEASE_SR = 20.0
# ② YIELD-TO-MOMENTUM: a focus whose last TWO readings gained >= YIELD_ENTER_PP combined while its
#   SR sits in [LEARNABLE_LO, graduate_sr) has been taken over by natural momentum -> WATCH (all
#   siege privileges withdraw; the watch registry keeps the notes). Resumes on stall (two consecutive
#   gains < YIELD_STALL_PP while still below graduate_sr); RESUME_LOCK readings of hysteresis prevent
#   the fix9-#2-style oscillation. Math guard: the trigger needs >= 7.5pp/reading — gnome_warrior-type
#   true walls (0.5-1.5pp/reading) can never hit it, so real sieges are untouched by construction.
YIELD_ENTER_PP = 15.0
LEARNABLE_LO = 20.0
YIELD_STALL_PP = 3.0
RESUME_LOCK_READINGS = 2
# ③ ZERO-WIN DISCOUNT: a focus with NO held-out win ever runs at half generation-seat share and a
#   per-session force-activation cap; the first win ratchets it to full price. Bounds the worst-case
#   monopoly tax of an unreachable wall (kobold: 22 sessions full price) without touching the fix8 ⑤
#   unlimited-duration rule for COMBAT (duration and burn-rate are different axes).
#   v6fix10.1 hazard-2: "no win" means SR has never exceeded ZERO_WIN_MAX_SR, not strictly zero —
#   held-out runs 1024 envs, so one fluke episode reads 0.098% and would permanently unlock full
#   price on a wall that is still effectively unreached (sewers-line skills sit at exactly 0.1%).
ZERO_WIN_SEAT_FRAC = 0.5
ZERO_WIN_FORCE_CAP = 2
ZERO_WIN_MAX_SR = 1.0
# ⑤ HIGH-WATER PROTECTION: any skill that ever held >= HIGHWATER_SR joins the protected set (pure
#   code ratchet, zero LLM); falling below high_water - HIGHWATER_DROP_PP flags it FORGETTING for
#   rehearsal. Fixes the fix8 erosion (iron_sword 67.8 -> 59.2 with nobody guarding it: the old
#   protected set covered only conquered/graduated walls and their links).
#   v6fix10.1 hazard-4: the ratchet needs TWO consecutive readings >= HIGHWATER_SR and records the
#   pair's MIN — a single lockhole-burst overshoot (open_chest 0->61 in one snapshot, settling at
#   ~45) must not poison the peak and put the skill into permanent phantom FORGETTING.
HIGHWATER_SR = 60.0
HIGHWATER_DROP_PP = 15.0

# --- v7 SPAWN-ANNEAL RELAY (v7_design.md §2/§5.5) --------------------------------------------------
# The multiplication chain (spawn->FloorN exposure = ∏ per-floor descent SR) makes ADVANCED-tier
# walls structurally unreachable from natural-spawn levels (baseline s151: all 24 ADVANCED+ = 0).
# A RELAY campaign attacks a deep wall backward: R0 spawns the student AT the target floor with a
# winners'-median kit (fix9 proved trained 95-100% is reachable there), then anneals the spawn point
# UP one floor per graduation until natural spawn (= held-out semantics) is reached ("sewn").
# The rung state machine is TRAINED-SR-driven (mid-rung levels contribute nothing to held-out by
# construction); readings must be FRESH (fix9 #2: session-stamped, never high-water). While a relay
# is active and un-sewn, the wall is EXEMPT from the gap gate / ④ attribution forcing / ⑤ enabler
# budget / ② yield (the rung machine carries its own progress accounting + early stop); all of them
# resume the moment the relay sews (spawn_floor == 0). R0's floor is proposed BY THE LLM
# (relay_r0_floor) — floor knowledge is legal teacher knowledge; this code never maps skills to
# floors, it only counts an integer down.
RELAY_MAX = 1                    # concurrent relay campaigns (fix9 monopoly tax -13: cap the opportunity cost)
RUNG_GRADUATE_SR = 70.0          # fresh TRAINED SR at/above this ...
RUNG_GRADUATE_CONSECUTIVE = 2    # ... this many consecutive readings -> spawn moves UP one floor
RUNG_FLOOR_SR = 20.0             # a fresh rung reading below this ...
RUNG_STALL_READINGS = 4          # ... this many consecutive readings -> regress one rung (re-consolidate)
RUNG_MOMENTUM_PP = 5.0           # new-high trained gain over the last 2 readings >= this = "progressing"
                                 # (③ discount reads THIS for relay walls, not held-out wins)
RELAY_STALL_PATIENCE = 3         # readings with neither a rung transition nor a trained new-high
                                 # (>= gap_stall_min_gain_pp) -> retire the campaign (COMBAT x2)

# --- v7fix3: relay unlock (P1-P3) + ecology economics (P5-P6) -------------------------------------
# v7fix2 post-mortem (jobs 3813092/3812896, 2026-07-10): pigman (tier-4) opened as an ORDINARY
# focus at s21 and triple-locked the relay until stall-retirement — (A1) no tier gate on the
# normal-open path, (A2) `kept()` swallowed any relay_r0_floor re-proposal for an active wall,
# (A3) the expand gate's "any focus >= 50%" condition blocked NEW relay campaigns behind a 0%
# focus. Meanwhile fix11's make_iron_armour focus (mid-band gear, never zero-win) force-activated
# 17-24 iron levels/session unthrottled and starved the whole INTERMEDIATE tier (BREADTH 5/276).
TIER4_RELAY_ONLY = True          # P1: a VERY_ADVANCED (tier-4) wall may open ONLY as a relay campaign
RELAY_ATTACH = True              # P2: re-proposing an ACTIVE zero-win focus WITH relay_r0_floor
                                 #     upgrades it in place (rung machine attached, stall state reset)
RELAY_EXPAND_EXEMPT = True       # P3: a relay open needs only a free slot + relay capacity — the
                                 #     "any existing focus >= expand_sr" condition is waived for it
FOCUS_FORCE_CAP = 8              # P5: per-wall per-session cap on FULL-PRICE force-activations
                                 #     (non-zero-win walls). Calibrated 2026-07-10: fix8's winning
                                 #     gnome phase ran 6-10/session (mean 7.2, natural top 10) —
                                 #     the cap sits above the healthy mean and only clips 10-streaks.
BREADTH_FRONTIER_START = 1       # P6: floor 1 is always unlocked for deep-spawn BREADTH levels
BREADTH_FRONTIER_SR = 60.0       # P6: trained SR at the frontier floor that unlocks the next floor
BREADTH_SPAWN_QUOTA = 6          # P6: max deep-spawn BREADTH levels kept per session (v7fix2's
                                 #     healthy dormant phase used 1-7/session, mean 3.7 — p85 ≈ 6)
# (v7fix3.1 audit: ZERO_WIN_UPGRADE_MAX_SR removed — dead constant; the P2 upgrade check reads
# th.zero_win_max_sr directly, same 1.0 value.)

# --- v7fix4: habitat fidelity (fable_research_reports/v7fix4真实世界接力与栖息地保真方案.md) ------
# v7fix3 post-mortem (job 3840016, killed @s61): lizard's relay ran its FULL lifecycle (attach s25
# -> graduated s55 -> SEWN s59) yet held-out stayed EXACTLY 0 (gap 97pp) — the FM-authored siege
# levels moved the lizard SHALLOWER along with the annealing spawn (smoking gun: trained SR ROSE
# 76->94->97 as the spawn moved UP), the modeler anchored R0 at floor 2 while lizards inhabit
# floor 3 (it is the one role without the mob floor table), and the fake in-level wins permanently
# disarmed ⑦ (zero-wins-ever keyed). These knobs anchor the campaign to the world's real habitat.
WALL_FLOOR_ANCHOR = True         # P1: R0 = the wall's native habitat floor (LLM's relay_r0_floor is
                                 #     advisory; mismatch -> r0_corrected) + entrance-chain autofill
DEEP_WALL_RELAY_FLOOR = 3        # P1: native floor >= this may open ONLY as a relay. Floor-3+ doors
                                 #     (enter_sewers and deeper) have NEVER opened naturally in any
                                 #     150-session arm incl. base; floor-2 walls keep the fix10
                                 #     door-gate route — fix8's winning gnome path stays untouched.
RELAY_KIT_STRIP = True           # P3: after the floor rungs complete, ONE final rung at natural
                                 #     spawn with an EMPTY kit (== held-out distribution); only its
                                 #     graduation SEWs — so SEWN is a result certificate, not a
                                 #     process milestone.
SANDBOX_MISMATCH_READINGS = 3    # P4 sentinel: a SEWN wall reading trained >= rung_graduate_sr with
                                 #     held-out still ~0 this many consecutive gap decisions retires
                                 #     as sandbox_mismatch (training dist != real dist — by
                                 #     construction unreachable once relay levels are system-built;
                                 #     this is the alarm for an unknown 4th degree of freedom).

# --- v7fix4.2: deep-wall relay TRIGGER (2026-07-12 hot patch) --------------------------------------
# fix4 run post-mortem @s77: the modeler NEVER proposed a floor-3+ wall in 77 sessions — zero
# deep_locked refusals, zero relay_r0_floor asks. Root cause is structural, not a mood: fix4's deep
# lock closed the one relay on-ramp any run had actually used (fix3's lizard: ordinary open ->
# zero-win readings -> ★ZERO-WIN hint -> in-place relay upgrade), leaving only a cold-start path
# ("propose WITH relay_r0_floor from day one") that no LLM has ever taken spontaneously, while
# auto-open deliberately excludes deep walls. The fix7 lesson replayed one level up: an entry the
# design HOPES the FM takes must be pinned in code, or it never happens. Three-layer repair:
#   (1) AUTOCONVERT — an ordinary proposal of a floor-3..(tier4-1) wall no longer bounces off
#       deep_locked; it is converted IN PLACE into a relay ask (relay_r0_floor := native floor) and
#       falls through to the EXISTING explicit-relay pipeline (⑦ exemption, expand exemption,
#       relay_max capacity + its fall-through closures, r0 anchoring — all reused, zero new paths).
#       decision: ``relay_converted``. tier-4 keeps the plain tier_locked refusal (payoff too far
#       to auto-commit the single relay slot); a wall with real held-out wins keeps deep_locked
#       (it needs no spawn ladder — same evidence rule as the attach path).
#   (2) JOURNAL TRIGGER — while (mature AND no live relay AND a free slot AND an eligible deep wall
#       exists), the journal renders a directive naming the eligible walls: "put ONE in your foci
#       this session; it auto-converts". The decision-moment concrete instruction (fix1 f2 lesson:
#       specific beats generic) replaces the old deterrent phrasing ("will be refused").
#   (3) K-SESSION FORCE — reproducibility backstop (from-scratch runs must not depend on the
#       LLM's mood or on a lucky journal narrative): if the trigger stays ignored for
#       RELAY_TRIGGER_FORCE_SESSIONS consecutive decisions, the system opens the top eligible
#       candidate itself (combat walls first, then shallower native floor, then name — the same
#       category boundary auto-open already uses). decision: ``relay_forced``. This is a bounded,
#       escalation-only exception to "auto-open never builds a relay" (v7_design.md): the LLM keeps
#       agency for K sessions every time, and the counter resets whenever it answers.
DEEP_WALL_AUTOCONVERT = True     # (1) ordinary deep-wall proposal converts to a relay ask in place
RELAY_TRIGGER_HINT = True        # (2) journal renders the deep-wall directive while conditions hold
RELAY_TRIGGER_FORCE_SESSIONS = 3 # (3) hint ignored this many consecutive decisions -> system opens

# --- v7fix4.5: attribution-driven succession + expand-gate relay exclusion (2026-07-13) -----------
# fix4.4 post-mortem @s114 (job 3936082): the kobold campaign ran at FULL dose with an honest
# anchor and still crawled at 9-14% — and the modeler's own VERIFIED attribution had already named
# the reason (chain_unreached, key_missing_link=enchant_sword: unenchanted iron barely damages a
# kobold; its style_note added learn/cast_fireball to the chain, and relay episodes were casting
# fireball 44% incidentally). But every road to ACTING on that diagnosis was locked: the equipment
# walls are deep-locked (relay-only) while the single relay slot was held by kobold, and the
# ordinary expand gate ("any focus >= 50%") was held hostage by the relay focus's
# by-construction-zero held-out SR (s112: expand_refused(defeat_gnome_warrior) — the modeler was
# trying to open exactly the combat sub-curriculum its tactic called for). fix7's lesson, third
# form: the mechanism must leave the FM a road to act on what it correctly diagnosed.
RELAY_SUCCESSION = True          # P1: a relay retiring with a VERIFIED chain_unreached attribution
                                 #     hands the trigger queue to its own chain's unmastered deep
                                 #     walls (tree order, shallow->deep) — equipment before re-fight
RELAY_EXPAND_EXCLUDED = True     # P2: the expand test (b) ignores ACTIVE RELAY foci (their held-out
                                 #     is 0 by construction through every mid rung); ordinary foci
                                 #     still gate each other — the anti-sprawl semantics survive

# --- v7fix4.6: descent-wall cliff-split sub-rungs + oscillation liveness + succession widening ----
# Double post-mortem @2026-07-13 (fast 3941260 s147-151 / 8100-replay 3940678 s112-114, fully
# reproduced on both arms): kobold's R0 (floor-3 point-blank) graduated CLEANLY at 73% — first rung
# graduation in v7 history — then the next rung (floor-2 spawn) read 0% flat. Craftax's descent is a
# COMPOUND gate (monsters_killed inits 0 for floors>0 -> the down ladder is LOCKED until 8 kills;
# spawn is at the up-ladder while the down-ladder is a random far tile: offline calibration over 80
# real worlds measured entry->down-ladder Manhattan P50=24-28, P90=38-49, walk-only-reachable as low
# as 41% on floor 3). One whole floor of descent = "clear 8 mobs + traverse a dark floor + descend"
# in a single rung increment: 73% -> 0% is a zero-success cliff, and 0% success = zero PPO gradient
# — not slow learning, structurally unlearnable. Three machine gaps compounded it: (a) the state
# machine's only moves were up/down/retire — regressing re-consolidates an already-73%-graduated
# rung, which adds nothing at a cliff; (b) LIVENESS HOLE: every transition reset best_rung_trained
# to None, so the first post-transition reading was always a "new high" and reset stall_patience —
# an oscillating ladder NEVER retired (the very case the fix9 #2 comment promised to burn patience
# on); (c) succession only accepted class==chain_unreached while both arms' VERIFIED diagnosis was
# execution_failure — even a retirement would have dropped the diagnosis on the floor. Repairs:
#   P1 CLIFF-SPLIT SUB-RUNGS — a 4th state-machine move. Two generic annealing dials, both existing
#      world_builder primitives: spawn distance to the floor's DOWN ladder (radius r, Manhattan —
#      the same metric as the placement masks) and a pre-credited monster count (the clear-gate
#      dial, set_monsters_killed). Sub-stage table (higher = easier), radii calibrated offline:
#      5: r<=1 credit 8 | 4: r<=8 credit 8 (one obs window, ~P10) | 3: r<=20 credit 8 (~P25-P40)
#      | 2: entry credit 8 | 1: entry credit 4 | 0 = FULL (entry, credit 0 — the pre-4.6 rung,
#      held-out semantics unchanged). A new floor still starts at FULL; a cliff (first
#      RUNG_CLIFF_READINGS readings all <= RUNG_CLIFF_SR, fires BEFORE the stall count) splits to
#      the easiest unmastered stage; scaffold stages graduate on x1 (a false advance wastes one
#      stage — the x2 philosophy stays on FULL, whose graduation IS the floor graduation); a
#      scaffold stall steps one stage easier; stalling at the easiest stage regresses the floor.
#      Per-floor resume memory (sub_stage_by_floor) so a regressed floor re-enters at the stage it
#      left, not at FULL. The KIT_STRIP exam never splits (it is the certificate); R0 splits since
#      v7fix4.7 Q1 (no clear-gate stages there — see the 4.7 block below).
#   P2 LIVENESS x2 — best_by_rung remembers each (floor, stage) rung's best across transitions
#      (revisits restore it: a cheap re-climb is no "new high", patience burns as fix9 #2 intended)
#      + RELAY_MAX_REGRESSIONS caps ALL regress-family moves per campaign; the cap retires through
#      the normal machinery so the succession can consume the archived attribution.
#   P3 SUCCESSION WIDENING — verified execution_failure with a named non-entrance key is exactly as
#      actionable as chain_unreached (the class boundary predates execution_failure carrying keys).
RUNG_CLIFF_SR = 5.0              # P1: a reading at/below this is "cliff evidence" (0-5% ~ no gradient)
RUNG_CLIFF_READINGS = 2          # P1: consecutive cliff readings at a fresh FULL rung -> SPLIT
RUNG_CLIFF_SPLIT = True          # P1: master switch (ablation / old-test isolation)
RUNG_SUBSTAGE_GRADUATE_X = 1     # P1: scaffold stages graduate on x1 (FULL keeps rung_graduate_consecutive)
RUNG_LADDER_RADII = (1, 8, 20)   # P1: spawn radius for stages 5/4/3 (offline-calibrated 2026-07-13:
                                 #     obs window half-extent 4-5; entry->ladder Manhattan P50=24-28)
RUNG_CLEAR_CREDIT_FULL = 8       # P1: stages 5..2 pre-credit (= MONSTERS_KILLED_TO_CLEAR_LEVEL, ladder open)
RUNG_CLEAR_CREDIT_HALF = 4       # P1: stage 1 pre-credit (kill 4 of 8 — half the clear gate)
RELAY_MAX_REGRESSIONS = 4        # P2: regress-family moves (floor + sub-stage + kit-strip restore)
                                 #     allowed per campaign; the next one retires instead
RELAY_SUCCESSION_CLASSES = ("chain_unreached", "execution_failure")  # P3

# --- v7fix4.7: R0 scaffold + DEFEND-driven relay patience + blacklist exemption --------------------
# Zero-shot arbitration @2026-07-14 (jobs 3966171/3966348: golden ckpt-8100, lr=0, the exact
# roll-1 task_800 R0 level, ~3000 episodes): TRUE zero-shot kobold SR = 0.28%. The 11-18% "first
# readings" of fix4.2/4.5 were WITHIN-SESSION PPO bootstrap (x50 in one session) seeded by ~2
# Poisson win-seeds per session; fix4.6's two rolls (0.5-2%, x1.4-1.5 per reading) were the SAME
# student whose dice missed — takeoff is a stochastic autocatalysis, not a code regression.
# Behavioural forensics (full 67-achievement breakdown, kit/floor RESET pre-credits stripped —
# collect_*/make_*/enter_* read ~100% as pure kit artifacts): the zero-shot student FIGHTS
# (fire_bow 84%), lights (place_torch 69%), walls up (place_stone 63%), sleeps underground
# (wake_up 60%) — but drifts UP to its trained comfort zone (defeat_gnome_warrior 31% = a
# floor-2 mob) instead of hunting the R0 target (kobold 0.28%, lizard 0.2%): off-target income
# outcompetes the unlearned fight. Repairs:
#   Q1 R0 SCAFFOLD — cliff-split may now fire AT R0 (the target floor) too: same generic dials
#      (spawn radius to the floor's down ladder — a fixed remote landmark, i.e. a LIT arena away
#      from the entry, 9x9 pre-light riding set_starting_floor). The clear-gate stages 2/1 are
#      DESCENT anneals, meaningless on the target floor: R0 climbs 5 -> 4 -> 3 -> 0 (FULL) in
#      both directions, and its scaffold NEVER emits a monster credit (the R0 down-gate stays
#      LOCKED so the campaign cannot leak below its target floor). FULL keeps held-out semantics.
#   Q2 DEFEND-DRIVEN PATIENCE — roll-1 died 2-3 readings short of escape velocity: readings
#      0.51 -> 0.74 -> 1.06 -> 1.58 -> 2.38 (x1.4-1.5 each, a textbook exponential climb) yet
#      every one burned patience because none cleared best +3pp; the escalation ladder's slope
#      channel never reached the relay counter. Repair — facts verified, never narratives (the
#      2026-07-14 lesson: the modeler's story AND the first eval read were both wrong; only the
#      reading sequence was true): a micro-ratchet log marks strict new absolute maxima (ANY
#      size). At patience exhaustion, if the ratchet rose >= RELAY_DEFEND_RISING_K of the last 3
#      readings and budget remains, ONE defence window opens instead of retirement: the modeler
#      must cite the actual recent readings in the focus's style_note (>= 2 of the last 3,
#      verified numerically against rung_trained); a verified citation resets patience and burns
#      one defence. Noise cannot sustain a rising ratchet (each fake high lifts the bar);
#      oscillation re-climbs sit below best_by_rung so their ratchet never rises — fix9 #2 and
#      the fix4.6 P2 liveness design are untouched.
#   Q3 BLACKLIST EXEMPTION — a retirement cut while the micro-ratchet was still rising is a slow
#      true climb stopped by a budget, not a failed tactic: cooldown + archive apply normally,
#      but it does NOT stack toward the 2-strikes blacklist (rising_retirements subtracts), so
#      succession/reopen stays alive for exactly the walls the patience knife mis-cut.
RUNG_R0_SCAFFOLD = True          # Q1: master switch — cliff-split may fire at spawn_floor == r0_floor
RELAY_DEFEND_BUDGET = 2          # Q2: defence windows per campaign
RELAY_DEFEND_RISING_K = 2        # Q2: micro-ratchet must rise >= K of the last 3 readings for a window

# --- v7fix5.3: DESCENT-REGIME scaffold (trap removal + clock anneal) -------------------------------
# Death-forensics probe @2026-07-16 (armA ckpt-15500, jobs 4031672/4046511, per-episode PHYSICAL
# state telemetry — the third attribution of this wall, and the first with location data): the
# stage3->stage2 collapse was NEVER "gnome exposure en route". 87% of stage-2 deaths happen ABOVE
# the rung floor — the student spawns on the entry (up-ladder) tile, abandons the descent within
# ~72 steps, climbs into the UNCLEARED floor above (3x spawn rate: only the rung floor is
# pre-credited) and dies there of sleep-kill (40%, energy collapse forces unbarricaded sleep) or
# thirst (32%, the mines have no water). Winners never need the survival loop at all: they cross
# in ~26 steps of a ~112-step speedrun — the resource clocks cannot ring on the winning path.
# The old ladder annealed pure geometry inside a trapped environment; fix5.3 removes the traps
# first, then re-runs the SAME radius dial inside the new regime (same dial, new meaning: pure
# navigation distance). Paired what-if arbitration (512 shared worlds, zero training):
# base 14.1% / needs0.3x 19.1% / uplock 21.3% / BOTH 25.0% — and uplock ALONE relocates the
# wander-sleep-death to the rung floor (55% sleeping), so the lock MUST ship with the slow clock.
# Two dials, both existing engine primitives, code-driven only (no FM authority — v7fix3 lesson):
#   UPLOCK — the rung floor's up-ladder ITEM is removed post-build (game_mechanics: ASCEND
#     requires standing on LADDER_UP): no escape leg, every death lands its gradient on-floor.
#     NEVER at FULL / kit-strip / held-out (exam semantics untouched).
#   NEEDS  — TaskParams.needs_depletion_multiplier anneals 0.3 -> 0.6 -> 1.0: the survival
#     clocks (hunger/thirst/fatigue) enter the episode window only after navigation is learned.
# New sub-stage table (higher = easier; stages 2/1/0 are the pre-5.3 stages 2/1/0 verbatim):
#   8: r<=1  credit 8 | 7: r<=8 credit 8 | 6: r<=20 credit 8   (the old 5/4/3, unchanged)
#   5: r<=20 credit 8 UPLOCK needs 0.3   (regime entry at a mastered distance)
#   4: entry credit 8 UPLOCK needs 0.3   (the what-if D condition, 25% zero-shot > 20% regress bar)
#   3: entry credit 8 UPLOCK needs 0.6   (clock-anneal leg)
#   2: entry credit 8                    (full clocks, escape open — the old stage 2 exactly)
#   1: entry credit 4 | 0 = FULL         (the old clear-gate leg, unchanged)
# R0 skips stages 5..1 in BOTH directions (descent-leg anneals, meaningless on the target floor —
# the fix4.7 Q1 rule extended to the new stages). Floor-1 pre-credit was PROVEN irrelevant under
# the lock (what-if E ≡ D bit-for-bit: locked trajectories never reach the floor above) — NOT
# shipped. The old 6-stage table stays pinned in pytest behind rung_descent_regime=False.
RUNG_DESCENT_REGIME = True       # master switch — False = exact pre-5.3 6-stage ladder
RUNG_NEEDS_SLOW = 0.3            # needs_depletion multiplier at stages 5/4
RUNG_NEEDS_MID = 0.6             # stage 3 anneal step back toward 1.0

# --- v7fix5.4: QUANTILE ladder — radius rungs from the floor's MEASURED distance distribution ------
# Radius-probe (2026-07-17, armA ckpt~17300, 512 paired worlds): the fix5.3 static radii stop at
# r<=20 while the entry distance distribution runs P50=22 / P90=40 — and zero-shot SR across
# r in {24,28,34,40} sits at 66..55% (ALL inside the fast-learning band), vs 30% at entry. The
# static ladder therefore under-samples exactly the distances entry needs (the old 73%->14%
# r<=20->entry transfer failure). Under the quantile ladder:
#   - the radius rungs are resolved BY CODE from RUNG_CALIB_SAMPLES freshly built worlds of the
#     actual floor (per-floor adaptive — the next floor's distribution is measured, not assumed;
#     the modeler never sets a radius, the v7fix3 law);
#   - EVERY radius rung carries the descent regime (uplock + slow clocks) so its training
#     distribution matches the entry target (the probe's dose-response was measured regime-on);
#   - R0 keeps its fix5.3 plain radius legs (regime knobs are descent-leg anneals — meaningless
#     on the target floor), and an UNRESOLVED floor renders the fix5.3 table byte-for-byte
#     (calibration lands at relay-build time before the first quantile build).
# Sub-stage table with Q resolved radii [q1<=..<=qQ] (higher stage = easier): max = 4+Q;
#   (4+Q)..5: r<=q1..qQ credit 8 UPLOCK needs 0.3 | 4/3/2/1/0: the fix5.3 entry leg verbatim.
# R0 skip set shrinks to stages 4..1 (the radius legs stay meaningful at R0, the entry
# anneals do not).
RUNG_QUANTILE_LADDER = False     # master switch (hotfix arm: siege_rung_quantile_ladder=true)
RUNG_LADDER_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.90)  # distance quantiles -> radii, easiest first
RUNG_CALIB_SAMPLES = 64          # worlds sampled per floor for the D2D measurement

# --- v7fix5.5 PROBE-AS-TOOL (design doc batch-2; user architecture 2026-07-17) --------------------
# The modeler may request ONE measurement per session on an ACTIVE RELAY wall; the code validates,
# budgets and stores it as probe_pending; the MAIN THREAD executes it between sessions on the
# current policy (rung_probe.py) and delivers the report for the NEXT journal page. The LLM's
# whole freedom = 4 fields (kind / filter / axis+direction / justification), each behind a code
# gate; every number is code-chosen. A probe only MEASURES: its output never writes training
# state (reports render as journal facts; the rung reading stream is untouched).
PROBE_KINDS = ("diagnose", "whatif")
PROBE_AXES = ("spawn_anchor", "radius", "pre_light", "monster_credit", "uplock", "needs_clock")
PROBE_DIRECTIONS = ("easier", "harder")
PROBE_FILTER_OPS = ("==", ">", "<")
# Sensor catalog: field -> (lo, hi) legal filter-value range. SINGLE SOURCE for filter validation,
# the journal's availability line, and the executor's snapshot dict (rung_probe derives its snap
# fields from THIS table's keys — a hand-maintained second copy would drift). It may GROW via the
# modeler's EVIDENCE GAP asks; it never silently shrinks.
PROBE_SENSORS: dict = {
    "health": (0, 100), "food": (0, 20), "drink": (0, 20), "energy": (0, 20),
    "sleeping": (0, 1), "floor": (0, 8),
    "melee_near": (0, 9999), "melee_c3": (0, 64), "melee_c8": (0, 64),
    "ranged_near": (0, 9999), "ranged_c3": (0, 64), "ranged_c8": (0, 64),
    "lava3x3": (0, 9), "ladder_dist": (0, 9999), "kills_gate": (-8, 100),
    "light": (0.0, 1.0),
    "inv_torches": (0, 99), "inv_arrows": (0, 99), "inv_stone": (0, 99),
    "inv_wood": (0, 99), "inv_potions": (0, 99),
}
PROBE_BUDGET_WINDOW = 10       # rolling budget window (sessions), per wall — verify kind
PROBE_BUDGET_WINDOW_FAST = 5   # v7fix5.6 (user 2026-07-18): diagnose/whatif window halved —
                               # honest zero-shot readings make real stalls last many sessions;
                               # one hypothesis iteration per 10 sessions was slower than the
                               # wall. verify keeps the 10-session window (B3.4 decision).
PROBE_BUDGET_PER_KIND = 1      # within the window: <= 1 diagnose AND <= 1 whatif per wall
PROBE_SNAPSHOT_K = 15          # raw episode-end snapshots delivered per report
PROBE_STALE_SESSIONS = 5       # report older than this at render time -> STALE banner

# --- v7fix5.5 P2 hypothesis loop (design doc batch-3): free attribution + in-machine verify ------
# A delivered probe report triggers ONE "scientist" LLM pass (think:on, small prompt) that files a
# ROOT-CAUSE HYPOTHESIS = {hypothesis (open vocabulary), evidence (must cite report/reading
# numbers, Tier-1 checked), intervention {axis, direction} (the SAME closed primitive menu as
# whatif probes — never a number), prediction}. On a stalled rung the intervention is verified by
# a paired whatif measurement (Tier-2); delta >= the bar COMPILES it into the ladder as an
# INSERTED rung the existing state machine fully governs (graduate -> back to the stalled stage;
# stall -> the insert is removed and the normal regress path runs — a wrong hypothesis self-heals,
# no LLM retraction needed). Categories are inferred, interfaces are pinned.
HYPOTHESIS_LOOP = True             # master switch (siege_hypothesis_loop)
HYPOTHESIS_VERIFY_DELTA_PP = 8.0   # paired zero-shot delta at/above this = VERIFIED -> compile
HYPOTHESIS_VERIFY_PER_WINDOW = 1   # Tier-2 verify probes per wall per PROBE_BUDGET_WINDOW
# The inserted rung's sub_stage id. A DISTINCT int far outside any ladder range (max = 4+Q) so the
# fix4.6 reading-isolation filter (exact stage match, gen_manager) isolates insert readings with
# ZERO filter changes, and best_by_rung keys stay the plain "floor:stage" scheme.
RUNG_INSERT_STAGE = 50

# v7fix5.7: the light-anneal leg of a pre-lit inserted rung. Graduating a pre_light=True insert
# does NOT fall straight back to the dark return stage (a one-shot ~-25pp context cliff — the
# fix5.4 measured effect and the fix5.5 verified +25.4pp, re-imposed at once): it first descends
# to THIS distinct stage id with the SAME knobs except pre_light="ladder" (the down ladder's 9x9
# stays torch-lit, the spawn stamp is removed — dark start, lit destination). Graduating the
# ladder-only leg pops the insert normally. A distinct id keeps the fix4.6 exact-stage reading
# isolation and a fresh best_by_rung key ("floor:49"), same scheme as the insert itself.
RUNG_INSERT_LIGHT_STAGE = 49

# --- v7fix5.7-P2' judgment statistics (fix56设计 §3; treats E4/E5) ---------------------------------
# LAW: judgment statistics must match the measurement's noise profile. The honest zero-shot
# source (fix5.6) is ±4pp heavy-tailed per reading — so EVERY rung judgment (new-high ratchet /
# patience / graduate / stall-regress) runs on the last-RUNG_WIN window MEAN of the current
# rung's raw readings. A not-yet-full window judges nothing (counters hold at 0): evidence,
# not absence, moves the machine. The raw per-reading series stays untouched in rung_trained —
# the DEFEND rising test reads THAT (§3.3), never a consumption-time subsequence.
RUNG_WIN = 3
RUNG_WIN_NEW_HIGH_PP = 2.0   # win3 new-high margin (replaces the single-read +3pp anchor:
                             # a lone lucky 40.4 can no longer set the patience anchor)

# --- §2① incremental success-experience recording (user 2026-07-05, 2nd round) --------------------
# CONQUERED is NOT a one-shot 80% retire gate anymore (conquered != saturated: a wall going 15%->40%
# is real progress worth recording). Instead, whenever a focus's SR rises by >= RECORD_DELTA_PP versus
# the LAST time it was recorded, we write/UPDATE (dedup by target) its success-experience entry. Noise
# jitter below this delta does NOT record -> no redundant near-duplicate entries. Focus retirement is
# governed purely by stall (FOCUS_MIN_STALL_SESSIONS), never by hitting an SR threshold.
RECORD_DELTA_PP = 10.0

# --- §2⑥ rehearsal FORGETTING prefilter thresholds (family-split, user 2026-07-05) ----------------
# A skill is a rehearsal candidate once it peaked >= *_MIN_PEAK then fell >= *_DROP_PP. COMBAT
# milestones peak LOW (~15%), so they get a low bar — else a forgetting combat milestone (exactly what
# §2.5 records and §3.6 must protect) would never be flagged. Gear/gather/explore keep the higher bar.
FORGETTING_MIN_PEAK = 40.0          # non-combat peak bar
FORGETTING_DROP_PP = 20.0           # non-combat drop bar
FORGETTING_COMBAT_MIN_PEAK = 15.0   # combat milestones peak low -> low bar (=combat record floor)
FORGETTING_COMBAT_DROP_PP = 10.0    # a 10pp slip off a low plateau is already a real slide


class SiegeThresholds:
    """All B-layer numeric thresholds for one SiegeNotebook, resolved from config (or defaults).

    Grouping them in one object keeps ``SiegeNotebook`` methods reading ``self.th.mastered_sr`` etc.
    instead of module globals, so a run's ``dicode_manager`` config can tune every knob. ``from_config``
    reads the ``siege_*`` keys off a config-manager object (any object with attributes / .get), falling
    back to the module-constant default for each missing key — so partial config or no config is fine.
    """

    __slots__ = (
        "mastered_sr", "unmastered_sr", "saturated_sr",
        "maturity_min_snapshots", "maturity_min_mastered", "maturity_skill_sr",
        "focus_improve_pp",
        # §2.6 multi-focus + §2① incremental recording (user 2026-07-05)
        "max_focus", "focus_expand_sr", "record_delta_pp",
        # §2⑥ family-split rehearsal FORGETTING prefilter (user 2026-07-05)
        "forgetting_min_peak", "forgetting_drop_pp",
        "forgetting_combat_min_peak", "forgetting_combat_drop_pp",
        # v6fix7 P1a escalation ladder + conquest
        "ladder_l1", "ladder_l2", "ladder_l3", "ladder_l4",
        "slope_window", "slope_min_pp",
        "cooldown_sessions", "blacklist_retirements", "conquest_consecutive",
        # v6fix7 P1c style_note lifecycle
        "note_stale_sessions",
        # v6fix8 ① fast-climb graduation / ③ gap hard gate / ⑤ enabler budget
        "graduate_sr", "graduate_consecutive", "maint_resiege_drop_pp",
        "gap_trained_min", "gap_min_pp", "gap_force_sessions",
        "enabler_max_sessions",
        # v6fix9 P3 gap early-stop
        "gap_stall_patience", "gap_stall_min_gain_pp",
        # v6fix10 reachability economics
        "door_min_sr", "yield_enter_pp", "learnable_lo", "yield_stall_pp",
        "resume_lock_readings", "zero_win_seat_frac", "zero_win_force_cap",
        "highwater_sr", "highwater_drop_pp",
        # v6fix10.1 hazard fixes
        "zero_win_max_sr", "gateway_release_sr",
        # v7 spawn-anneal relay
        "relay_max", "rung_graduate_sr", "rung_graduate_consecutive",
        "rung_floor_sr", "rung_stall_readings", "rung_momentum_pp", "relay_stall_patience",
        # v7fix3 relay unlock + ecology economics
        "tier4_relay_only", "relay_attach", "relay_expand_exempt",
        "focus_force_cap", "breadth_frontier_sr", "breadth_spawn_quota",
        # v7fix4 habitat fidelity
        "wall_floor_anchor", "deep_wall_relay_floor", "relay_kit_strip",
        "sandbox_mismatch_readings",
        # v7fix4.2 deep-wall relay trigger
        "deep_wall_autoconvert", "relay_trigger_hint", "relay_trigger_force_sessions",
        # v7fix4.5 attribution-driven succession + expand-gate relay exclusion
        "relay_succession", "relay_expand_excluded",
        # v7fix4.6 cliff-split sub-rungs + oscillation liveness
        "rung_cliff_sr", "rung_cliff_readings", "rung_cliff_split",
        "rung_substage_graduate_x", "rung_ladder_radii",
        "rung_clear_credit_full", "rung_clear_credit_half", "relay_max_regressions",
        # v7fix4.7 R0 scaffold + DEFEND-driven relay patience
        "rung_r0_scaffold", "relay_defend_budget", "relay_defend_rising_k",
        # v7fix5.3 descent-regime scaffold (uplock + needs-clock anneal)
        "rung_descent_regime", "rung_needs_slow", "rung_needs_mid",
        # v7fix5.4 quantile ladder (code-calibrated per-floor radii)
        "rung_quantile_ladder", "rung_ladder_quantiles", "rung_calib_samples",
        # v7fix5.5 P2 hypothesis loop
        "hypothesis_loop", "hypothesis_verify_delta_pp",
    )

    def __init__(
        self,
        mastered_sr=MASTERED_SR,
        unmastered_sr=UNMASTERED_SR,
        saturated_sr=None,
        maturity_min_snapshots=MATURITY_MIN_SNAPSHOTS,
        maturity_min_mastered=MATURITY_MIN_MASTERED,
        maturity_skill_sr=MATURITY_SKILL_SR,
        focus_improve_pp=FOCUS_IMPROVE_PP,
        max_focus=MAX_FOCUS,
        focus_expand_sr=FOCUS_EXPAND_SR,
        record_delta_pp=RECORD_DELTA_PP,
        forgetting_min_peak=FORGETTING_MIN_PEAK,
        forgetting_drop_pp=FORGETTING_DROP_PP,
        forgetting_combat_min_peak=FORGETTING_COMBAT_MIN_PEAK,
        forgetting_combat_drop_pp=FORGETTING_COMBAT_DROP_PP,
        ladder_l1=LADDER_L1,
        ladder_l2=LADDER_L2,
        ladder_l3=LADDER_L3,
        ladder_l4=LADDER_L4,
        slope_window=SLOPE_WINDOW,
        slope_min_pp=SLOPE_MIN_PP,
        cooldown_sessions=COOLDOWN_SESSIONS,
        blacklist_retirements=BLACKLIST_RETIREMENTS,
        conquest_consecutive=CONQUEST_CONSECUTIVE,
        note_stale_sessions=NOTE_STALE_SESSIONS,
        graduate_sr=GRADUATE_SR,
        graduate_consecutive=GRADUATE_CONSECUTIVE,
        maint_resiege_drop_pp=MAINT_RESIEGE_DROP_PP,
        gap_trained_min=GAP_TRAINED_MIN,
        gap_min_pp=GAP_MIN_PP,
        gap_force_sessions=GAP_FORCE_SESSIONS,
        enabler_max_sessions=ENABLER_MAX_SESSIONS,
        gap_stall_patience=GAP_STALL_PATIENCE,
        gap_stall_min_gain_pp=GAP_STALL_MIN_GAIN_PP,
        door_min_sr=DOOR_MIN_SR,
        yield_enter_pp=YIELD_ENTER_PP,
        learnable_lo=LEARNABLE_LO,
        yield_stall_pp=YIELD_STALL_PP,
        resume_lock_readings=RESUME_LOCK_READINGS,
        zero_win_seat_frac=ZERO_WIN_SEAT_FRAC,
        zero_win_force_cap=ZERO_WIN_FORCE_CAP,
        highwater_sr=HIGHWATER_SR,
        highwater_drop_pp=HIGHWATER_DROP_PP,
        zero_win_max_sr=ZERO_WIN_MAX_SR,
        gateway_release_sr=GATEWAY_RELEASE_SR,
        relay_max=RELAY_MAX,
        rung_graduate_sr=RUNG_GRADUATE_SR,
        rung_graduate_consecutive=RUNG_GRADUATE_CONSECUTIVE,
        rung_floor_sr=RUNG_FLOOR_SR,
        rung_stall_readings=RUNG_STALL_READINGS,
        rung_momentum_pp=RUNG_MOMENTUM_PP,
        relay_stall_patience=RELAY_STALL_PATIENCE,
        tier4_relay_only=TIER4_RELAY_ONLY,
        relay_attach=RELAY_ATTACH,
        relay_expand_exempt=RELAY_EXPAND_EXEMPT,
        focus_force_cap=FOCUS_FORCE_CAP,
        breadth_frontier_sr=BREADTH_FRONTIER_SR,
        breadth_spawn_quota=BREADTH_SPAWN_QUOTA,
        wall_floor_anchor=WALL_FLOOR_ANCHOR,
        deep_wall_relay_floor=DEEP_WALL_RELAY_FLOOR,
        relay_kit_strip=RELAY_KIT_STRIP,
        sandbox_mismatch_readings=SANDBOX_MISMATCH_READINGS,
        deep_wall_autoconvert=DEEP_WALL_AUTOCONVERT,
        relay_trigger_hint=RELAY_TRIGGER_HINT,
        relay_trigger_force_sessions=RELAY_TRIGGER_FORCE_SESSIONS,
        relay_succession=RELAY_SUCCESSION,
        relay_expand_excluded=RELAY_EXPAND_EXCLUDED,
        rung_cliff_sr=RUNG_CLIFF_SR,
        rung_cliff_readings=RUNG_CLIFF_READINGS,
        rung_cliff_split=RUNG_CLIFF_SPLIT,
        rung_substage_graduate_x=RUNG_SUBSTAGE_GRADUATE_X,
        rung_ladder_radii=RUNG_LADDER_RADII,
        rung_clear_credit_full=RUNG_CLEAR_CREDIT_FULL,
        rung_clear_credit_half=RUNG_CLEAR_CREDIT_HALF,
        relay_max_regressions=RELAY_MAX_REGRESSIONS,
        rung_r0_scaffold=RUNG_R0_SCAFFOLD,
        relay_defend_budget=RELAY_DEFEND_BUDGET,
        relay_defend_rising_k=RELAY_DEFEND_RISING_K,
        rung_descent_regime=RUNG_DESCENT_REGIME,
        rung_needs_slow=RUNG_NEEDS_SLOW,
        rung_needs_mid=RUNG_NEEDS_MID,
        rung_quantile_ladder=RUNG_QUANTILE_LADDER,
        rung_ladder_quantiles=RUNG_LADDER_QUANTILES,
        rung_calib_samples=RUNG_CALIB_SAMPLES,
        hypothesis_loop=HYPOTHESIS_LOOP,
        hypothesis_verify_delta_pp=HYPOTHESIS_VERIFY_DELTA_PP,
    ):
        self.mastered_sr = float(mastered_sr)
        self.unmastered_sr = float(unmastered_sr)
        # saturated now defaults to the module SATURATED_SR (80, user 2026-07-05), NOT mastered_sr.
        self.saturated_sr = float(saturated_sr if saturated_sr is not None else SATURATED_SR)
        self.maturity_min_snapshots = int(maturity_min_snapshots)
        self.maturity_min_mastered = int(maturity_min_mastered)
        self.maturity_skill_sr = float(maturity_skill_sr)
        self.focus_improve_pp = float(focus_improve_pp)
        # §2.6 multi-focus + §2① incremental recording.
        self.max_focus = int(max_focus)
        self.focus_expand_sr = float(focus_expand_sr)
        self.record_delta_pp = float(record_delta_pp)
        # §2⑥ family-split forgetting prefilter.
        self.forgetting_min_peak = float(forgetting_min_peak)
        self.forgetting_drop_pp = float(forgetting_drop_pp)
        self.forgetting_combat_min_peak = float(forgetting_combat_min_peak)
        self.forgetting_combat_drop_pp = float(forgetting_combat_drop_pp)
        # v6fix7 P1a escalation ladder + conquest.
        self.ladder_l1 = int(ladder_l1)
        self.ladder_l2 = int(ladder_l2)
        self.ladder_l3 = int(ladder_l3)
        self.ladder_l4 = int(ladder_l4)
        self.slope_window = int(slope_window)
        self.slope_min_pp = float(slope_min_pp)
        self.cooldown_sessions = int(cooldown_sessions)
        self.blacklist_retirements = int(blacklist_retirements)
        self.conquest_consecutive = int(conquest_consecutive)
        self.note_stale_sessions = int(note_stale_sessions)
        # v6fix8 ① graduation / ③ gap gate / ⑤ enabler budget.
        self.graduate_sr = float(graduate_sr)
        self.graduate_consecutive = int(graduate_consecutive)
        self.maint_resiege_drop_pp = float(maint_resiege_drop_pp)
        self.gap_trained_min = float(gap_trained_min)
        self.gap_min_pp = float(gap_min_pp)
        self.gap_force_sessions = int(gap_force_sessions)
        self.enabler_max_sessions = int(enabler_max_sessions)
        # v6fix9 P3 gap early-stop.
        self.gap_stall_patience = int(gap_stall_patience)
        self.gap_stall_min_gain_pp = float(gap_stall_min_gain_pp)
        # v6fix10 reachability economics.
        self.door_min_sr = float(door_min_sr)
        self.yield_enter_pp = float(yield_enter_pp)
        self.learnable_lo = float(learnable_lo)
        self.yield_stall_pp = float(yield_stall_pp)
        self.resume_lock_readings = int(resume_lock_readings)
        self.zero_win_seat_frac = float(zero_win_seat_frac)
        self.zero_win_force_cap = int(zero_win_force_cap)
        self.highwater_sr = float(highwater_sr)
        self.highwater_drop_pp = float(highwater_drop_pp)
        self.zero_win_max_sr = float(zero_win_max_sr)
        self.gateway_release_sr = float(gateway_release_sr)
        # v7 spawn-anneal relay.
        self.relay_max = int(relay_max)
        self.rung_graduate_sr = float(rung_graduate_sr)
        self.rung_graduate_consecutive = int(rung_graduate_consecutive)
        self.rung_floor_sr = float(rung_floor_sr)
        self.rung_stall_readings = int(rung_stall_readings)
        self.rung_momentum_pp = float(rung_momentum_pp)
        self.relay_stall_patience = int(relay_stall_patience)
        # v7fix3 relay unlock + ecology economics.
        self.tier4_relay_only = bool(tier4_relay_only)
        self.relay_attach = bool(relay_attach)
        self.relay_expand_exempt = bool(relay_expand_exempt)
        self.focus_force_cap = int(focus_force_cap)
        self.breadth_frontier_sr = float(breadth_frontier_sr)
        self.breadth_spawn_quota = int(breadth_spawn_quota)
        # v7fix4 habitat fidelity.
        self.wall_floor_anchor = bool(wall_floor_anchor)
        self.deep_wall_relay_floor = int(deep_wall_relay_floor)
        self.relay_kit_strip = bool(relay_kit_strip)
        self.sandbox_mismatch_readings = int(sandbox_mismatch_readings)
        self.deep_wall_autoconvert = bool(deep_wall_autoconvert)
        self.relay_trigger_hint = bool(relay_trigger_hint)
        self.relay_trigger_force_sessions = int(relay_trigger_force_sessions)
        # v7fix4.5 attribution-driven succession + expand-gate relay exclusion.
        self.relay_succession = bool(relay_succession)
        self.relay_expand_excluded = bool(relay_expand_excluded)
        # v7fix4.6 cliff-split sub-rungs + oscillation liveness.
        self.rung_cliff_sr = float(rung_cliff_sr)
        self.rung_cliff_readings = int(rung_cliff_readings)
        self.rung_cliff_split = bool(rung_cliff_split)
        self.rung_substage_graduate_x = int(rung_substage_graduate_x)
        self.rung_ladder_radii = tuple(int(r) for r in rung_ladder_radii)
        self.rung_clear_credit_full = int(rung_clear_credit_full)
        self.rung_clear_credit_half = int(rung_clear_credit_half)
        self.relay_max_regressions = int(relay_max_regressions)
        # v7fix4.7 R0 scaffold + DEFEND-driven relay patience.
        self.rung_r0_scaffold = bool(rung_r0_scaffold)
        self.relay_defend_budget = int(relay_defend_budget)
        self.relay_defend_rising_k = int(relay_defend_rising_k)
        # v7fix5.3 descent-regime scaffold.
        self.rung_descent_regime = bool(rung_descent_regime)
        self.rung_needs_slow = float(rung_needs_slow)
        self.rung_needs_mid = float(rung_needs_mid)
        self.rung_quantile_ladder = bool(rung_quantile_ladder)
        self.rung_ladder_quantiles = tuple(float(q) for q in rung_ladder_quantiles)
        self.rung_calib_samples = int(rung_calib_samples)
        # v7fix5.5 P2 hypothesis loop.
        self.hypothesis_loop = bool(hypothesis_loop)
        self.hypothesis_verify_delta_pp = float(hypothesis_verify_delta_pp)

    @classmethod
    def from_config(cls, dm) -> "SiegeThresholds":
        """Build from a dicode_manager config object; each missing ``siege_*`` key keeps its default."""
        def g(key, default):
            if dm is None:
                return default
            # Prefer .get (OmegaConf DictConfig / plain dict both support it, and it won't raise on a
            # missing key the way OmegaConf attribute access does).
            getter = getattr(dm, "get", None)
            if callable(getter):
                try:
                    val = getter(key, default)
                    return val if val is not None else default
                except Exception:  # noqa: BLE001
                    return default
            val = getattr(dm, key, default)
            return val if val is not None else default
        return cls(
            mastered_sr=g("siege_mastered_sr", MASTERED_SR),
            unmastered_sr=g("siege_unmastered_sr", UNMASTERED_SR),
            saturated_sr=g("siege_saturated_sr", None),
            maturity_min_snapshots=g("siege_maturity_min_snapshots", MATURITY_MIN_SNAPSHOTS),
            maturity_min_mastered=g("siege_maturity_min_mastered", MATURITY_MIN_MASTERED),
            maturity_skill_sr=g("siege_maturity_skill_sr", MATURITY_SKILL_SR),
            focus_improve_pp=g("siege_focus_improve_pp", FOCUS_IMPROVE_PP),
            max_focus=g("siege_max_focus", MAX_FOCUS),
            focus_expand_sr=g("siege_focus_expand_sr", FOCUS_EXPAND_SR),
            record_delta_pp=g("siege_record_delta_pp", RECORD_DELTA_PP),
            forgetting_min_peak=g("siege_forgetting_min_peak", FORGETTING_MIN_PEAK),
            forgetting_drop_pp=g("siege_forgetting_drop_pp", FORGETTING_DROP_PP),
            forgetting_combat_min_peak=g("siege_forgetting_combat_min_peak", FORGETTING_COMBAT_MIN_PEAK),
            forgetting_combat_drop_pp=g("siege_forgetting_combat_drop_pp", FORGETTING_COMBAT_DROP_PP),
            ladder_l1=g("siege_ladder_l1", LADDER_L1),
            ladder_l2=g("siege_ladder_l2", LADDER_L2),
            ladder_l3=g("siege_ladder_l3", LADDER_L3),
            ladder_l4=g("siege_ladder_l4", LADDER_L4),
            slope_window=g("siege_slope_window", SLOPE_WINDOW),
            slope_min_pp=g("siege_slope_min_pp", SLOPE_MIN_PP),
            cooldown_sessions=g("siege_cooldown_sessions", COOLDOWN_SESSIONS),
            blacklist_retirements=g("siege_blacklist_retirements", BLACKLIST_RETIREMENTS),
            conquest_consecutive=g("siege_conquest_consecutive", CONQUEST_CONSECUTIVE),
            note_stale_sessions=g("siege_note_stale_sessions", NOTE_STALE_SESSIONS),
            graduate_sr=g("siege_graduate_sr", GRADUATE_SR),
            graduate_consecutive=g("siege_graduate_consecutive", GRADUATE_CONSECUTIVE),
            maint_resiege_drop_pp=g("siege_maint_resiege_drop_pp", MAINT_RESIEGE_DROP_PP),
            gap_trained_min=g("siege_gap_trained_min", GAP_TRAINED_MIN),
            gap_min_pp=g("siege_gap_min_pp", GAP_MIN_PP),
            gap_force_sessions=g("siege_gap_force_sessions", GAP_FORCE_SESSIONS),
            enabler_max_sessions=g("siege_enabler_max_sessions", ENABLER_MAX_SESSIONS),
            gap_stall_patience=g("siege_gap_stall_patience", GAP_STALL_PATIENCE),
            gap_stall_min_gain_pp=g("siege_gap_stall_min_gain_pp", GAP_STALL_MIN_GAIN_PP),
            door_min_sr=g("siege_door_min_sr", DOOR_MIN_SR),
            yield_enter_pp=g("siege_yield_enter_pp", YIELD_ENTER_PP),
            learnable_lo=g("siege_learnable_lo", LEARNABLE_LO),
            yield_stall_pp=g("siege_yield_stall_pp", YIELD_STALL_PP),
            resume_lock_readings=g("siege_resume_lock_readings", RESUME_LOCK_READINGS),
            zero_win_seat_frac=g("siege_zero_win_seat_frac", ZERO_WIN_SEAT_FRAC),
            zero_win_force_cap=g("siege_zero_win_force_cap", ZERO_WIN_FORCE_CAP),
            highwater_sr=g("siege_highwater_sr", HIGHWATER_SR),
            highwater_drop_pp=g("siege_highwater_drop_pp", HIGHWATER_DROP_PP),
            zero_win_max_sr=g("siege_zero_win_max_sr", ZERO_WIN_MAX_SR),
            gateway_release_sr=g("siege_gateway_release_sr", GATEWAY_RELEASE_SR),
            relay_max=g("siege_relay_max", RELAY_MAX),
            rung_graduate_sr=g("siege_rung_graduate_sr", RUNG_GRADUATE_SR),
            rung_graduate_consecutive=g("siege_rung_graduate_consecutive", RUNG_GRADUATE_CONSECUTIVE),
            rung_floor_sr=g("siege_rung_floor_sr", RUNG_FLOOR_SR),
            rung_stall_readings=g("siege_rung_stall_readings", RUNG_STALL_READINGS),
            rung_momentum_pp=g("siege_rung_momentum_pp", RUNG_MOMENTUM_PP),
            relay_stall_patience=g("siege_relay_stall_patience", RELAY_STALL_PATIENCE),
            tier4_relay_only=g("siege_tier4_relay_only", TIER4_RELAY_ONLY),
            relay_attach=g("siege_relay_attach", RELAY_ATTACH),
            relay_expand_exempt=g("siege_relay_expand_exempt", RELAY_EXPAND_EXEMPT),
            focus_force_cap=g("siege_focus_force_cap", FOCUS_FORCE_CAP),
            breadth_frontier_sr=g("siege_breadth_frontier_sr", BREADTH_FRONTIER_SR),
            breadth_spawn_quota=g("siege_breadth_spawn_quota", BREADTH_SPAWN_QUOTA),
            wall_floor_anchor=g("siege_wall_floor_anchor", WALL_FLOOR_ANCHOR),
            deep_wall_relay_floor=g("siege_deep_wall_relay_floor", DEEP_WALL_RELAY_FLOOR),
            relay_kit_strip=g("siege_relay_kit_strip", RELAY_KIT_STRIP),
            sandbox_mismatch_readings=g(
                "siege_sandbox_mismatch_readings", SANDBOX_MISMATCH_READINGS
            ),
            deep_wall_autoconvert=g("siege_deep_wall_autoconvert", DEEP_WALL_AUTOCONVERT),
            relay_trigger_hint=g("siege_relay_trigger_hint", RELAY_TRIGGER_HINT),
            relay_trigger_force_sessions=g(
                "siege_relay_trigger_force_sessions", RELAY_TRIGGER_FORCE_SESSIONS
            ),
            relay_succession=g("siege_relay_succession", RELAY_SUCCESSION),
            relay_expand_excluded=g("siege_relay_expand_excluded", RELAY_EXPAND_EXCLUDED),
            rung_cliff_sr=g("siege_rung_cliff_sr", RUNG_CLIFF_SR),
            rung_cliff_readings=g("siege_rung_cliff_readings", RUNG_CLIFF_READINGS),
            rung_cliff_split=g("siege_rung_cliff_split", RUNG_CLIFF_SPLIT),
            rung_substage_graduate_x=g(
                "siege_rung_substage_graduate_x", RUNG_SUBSTAGE_GRADUATE_X
            ),
            rung_ladder_radii=g("siege_rung_ladder_radii", RUNG_LADDER_RADII),
            rung_clear_credit_full=g("siege_rung_clear_credit_full", RUNG_CLEAR_CREDIT_FULL),
            rung_clear_credit_half=g("siege_rung_clear_credit_half", RUNG_CLEAR_CREDIT_HALF),
            relay_max_regressions=g("siege_relay_max_regressions", RELAY_MAX_REGRESSIONS),
            rung_r0_scaffold=g("siege_rung_r0_scaffold", RUNG_R0_SCAFFOLD),
            relay_defend_budget=g("siege_relay_defend_budget", RELAY_DEFEND_BUDGET),
            relay_defend_rising_k=g("siege_relay_defend_rising_k", RELAY_DEFEND_RISING_K),
            rung_descent_regime=g("siege_rung_descent_regime", RUNG_DESCENT_REGIME),
            rung_needs_slow=g("siege_rung_needs_slow", RUNG_NEEDS_SLOW),
            rung_needs_mid=g("siege_rung_needs_mid", RUNG_NEEDS_MID),
            rung_quantile_ladder=g("siege_rung_quantile_ladder", RUNG_QUANTILE_LADDER),
            rung_ladder_quantiles=g("siege_rung_ladder_quantiles", RUNG_LADDER_QUANTILES),
            rung_calib_samples=g("siege_rung_calib_samples", RUNG_CALIB_SAMPLES),
            hypothesis_loop=g("siege_hypothesis_loop", HYPOTHESIS_LOOP),
            hypothesis_verify_delta_pp=g(
                "siege_hypothesis_verify_delta_pp", HYPOTHESIS_VERIFY_DELTA_PP
            ),
        )


# Mastery-flag vocabulary attached to every prereq-tree link.
LINK_STATES = ("CONSOLIDATED", "RISING", "UNMASTERED", "UNKNOWN")


def mastery_from_sr(
    sr: float | None,
    mastered_sr: float = MASTERED_SR,
    unmastered_sr: float = UNMASTERED_SR,
) -> str:
    """Map a held-out SR (0..100) to a link mastery flag by the B-layer thresholds.

    Single source of truth for "consolidated vs unmastered" — the LLM's self-report is never trusted.
    Thresholds default to the module constants but callers (SiegeNotebook) pass their config-resolved
    values so the flags honour a tuned MASTERED_SR/UNMASTERED_SR.
    """
    if sr is None:
        return "UNKNOWN"
    if sr >= mastered_sr:
        return "CONSOLIDATED"
    if sr <= unmastered_sr:
        return "UNMASTERED"
    return "RISING"


def _empty_notebook() -> dict:
    """The guaranteed-valid empty notebook schema (session 0, nothing sieged yet).

    §2.6 (user 2026-07-05): the single top-level ``focus`` became a ``foci`` LIST (up to MAX_FOCUS
    parallel sieges). Each focus element carries its OWN per-focus bookkeeping + prereq tree:
        {"skill", "started_session", "best_sr",
         "last_recorded_sr",          # §2① SR at the last success-experience record (dedup/delta base)
         "prereq_tree": [ {"skill","state","sr","role"}, ... ]}
    """
    return {
        "foci": [],                  # list of focus dicts (see docstring); empty = no active siege
        "verified_chains": [],       # §2.5 success-experience entries (dedup by target; categorised)
        "protected_set": [],         # skill names to rehearse (§3.6; v6fix7: VERIFIED conquests only)
        "last_session": None,        # last session index this notebook was updated at
        "history": [],               # append-only log of focus changes, for offline inspection
        # v6fix7 P1a: retirement registry — skill -> {count, last_session, sr_at_retirement,
        # link_sr_at_retirement, failed_notes[<=3]}. Drives cooldown / blacklist / "what's different
        # this time" checks in _reconcile_foci, and is rendered to the modeler (history made READABLE).
        "retired": {},
        # v6fix8 ①: maintenance registry — skill -> {graduated_session, sr_at_graduation, links,
        # style_note}. A graduated (fast-climb) wall: no siege privileges, rehearsal holds it, and
        # _reconcile_foci refuses re-siege until it truly collapses (< graduate_sr - resiege_drop).
        "maintenance": {},
        # v7fix5.5 PROBE-AS-TOOL state (in the _coerce whitelist — the fix4.2 lesson). ledger:
        # wall -> [[session, kind], ...] (rolling budget); pending: the ONE validated request
        # awaiting main-thread execution; reports: wall -> latest delivered report (rendered to
        # the journal, STALE-stamped when old); receipt: last accept/reject line (LLM-visible).
        "probe_ledger": {},
        "probe_pending": None,
        "probe_reports": {},
        # v7fix5.6 HONEST rung readings: wall -> the latest between-session zero-shot eval
        # {"session","sr","spawn_floor","sub_stage","n_envs"}, written by the MAIN thread
        # (rung_probe.run_rung_eval) and read by gen_manager as the ONLY number fed to
        # note_rung_reading for relay walls. Measurement decoupled from training (the fix56
        # law: entry trained-SR 43% vs zero-shot 24%, probe 2026-07-18). _coerce keeps it.
        "rung_eval": {},
        "probe_receipt": None,
        # v7fix5.5 P2: append-only ROOT-CAUSE HYPOTHESIS ledger — every scientist-pass output
        # with its full verdict lifecycle (rejected_tier1 / recorded / verify_scheduled /
        # verified_compiled / refuted / insert_graduated / insert_stalled / stale_context /
        # expired). The paper's hypothesis-quality curve reads straight off this key.
        "hypothesis_log": [],
        # v7fix4.2: deep-wall relay trigger state — {"armed", "ignored", "candidates"}. Persisted
        # (resume-safe) so the K-session force counter survives a checkpoint resume; _coerce drops
        # any key not listed here, which is why it must appear in this schema.
        "relay_trigger": {"armed": False, "ignored": 0, "candidates": []},
        # v6fix10 ②: watch registry — skill -> the full parked focus dict (+watch_since). A yielded
        # focus keeps its notes/chain tracking here but holds NO siege privileges (it is not in
        # ``foci``, so seats / force-activation / training quota / gap feeding all skip it by
        # construction) and does NOT occupy a MAX_FOCUS slot. Resumes into foci on stall.
        "watch": {},
        # v6fix10 ⑤: high-water registry — skill -> peak held-out SR ever observed at/above
        # HIGHWATER_SR. Pure code ratchet; protected_set() unions these in, and
        # highwater_forgetting() flags any that fell HIGHWATER_DROP_PP below their peak.
        "highwater": {},
        # v6fix10.1 hazard-4: pending high-water candidates — skill -> [session, reading]. The
        # ratchet confirms only on TWO consecutive readings >= HIGHWATER_SR (records their MIN).
        "highwater_pending": {},
        # v6fix10.1 hazard-3a: admission waiting room — skill -> {session, links}. A candidate wall
        # with no failure forensics yet is chain-tracked here (chain_targets() feeds it to the
        # ChainOrderLog) instead of being opened blind past the door gate.
        "pending_track": {},
        # v7fix3 P6: breadth spawn frontier — the deepest floor a non-relay BREADTH level may
        # declare as spawn_floor (monotone, floor 1 always unlocked; _coerce keeps only schema
        # keys, so the frontier must live here to survive a reload/resume).
        "breadth_frontier": BREADTH_FRONTIER_START,
        # v7fix3 P1/P0: tier-4 walls refused as ordinary sieges LAST session — mirrored into the
        # journal because focus-decision strings are LOG-ONLY (the ⑦ lesson: a refusal the LLM
        # never sees teaches nothing; ⑦ has its chain-hint channel, tier_locked gets this one).
        "tier_locked_last": [],
        # v7fix5.0 P1/P2: access-cap verdicts — {"session": idx, "caps": {wall -> access dict from
        # ChainOrderLog.access_frontier}}. Fed by gen_manager every session (note_access_caps).
        # Drives the gap-gate ACCESS_CAPPED park (P2), the watch-resume hold, and the
        # expand-gate frontier exemption. Persisted (resume-safe); _coerce drops unlisted keys,
        # which is why it must appear in this schema.
        "access_caps": {"session": None, "caps": {}},
        # v7fix5.4: per-floor entry->down-ladder distance quantile radii, MEASURED by gen_manager
        # at relay-build time (floor str -> [int radii, easiest first]). Code-owned — the modeler
        # never writes here (v7fix3 law). _coerce drops unlisted keys (the fix4.2 lesson), which
        # is why it must appear in this schema.
        "floor_d2d_radii": {},
    }


def _empty_focus(skill: str, session_idx: int, sr: float | None, opened_by: str = "llm") -> dict:
    """A fresh focus element for ``skill`` newly opened at ``session_idx``."""
    return {
        "skill": skill,
        "started_session": session_idx,
        "best_sr": sr,
        "last_recorded_sr": None,    # §2① no success-experience recorded for it yet
        "prereq_tree": [],           # backtracked chain, filled by the LLM proposal + code flags
        "style_note": "",            # §3.1 self-style: LLM free-text attack know-how for this wall
        # v6fix7 P1a ladder state:
        "sr_history": [] if sr is None else [float(sr)],  # recent focus SR readings (slope signal)
        "link_best": {},             # per-link best SR seen (foundation-progress signal)
        "frozen_sessions": 0,        # consecutive whole-tree no-progress sessions (drives the ladder)
        "ladder_level": 0,           # 0 none / 1 defend-or-switch / 2 forced form / 3 forced tactic
        "last_siege_type": "",       # TYPE of the latest siege level built for this wall (from level_meta)
        "consecutive_mastered": 0,   # consecutive snapshots at/above mastered_sr (conquest, #8 fix)
        # v6fix7 P1c style_note lifecycle (AutoManual-lite):
        "note_status": "active",     # active | stale (long unsupported) | contradicted (must rewrite)
        "note_last_supported_session": None,  # last session evidence_check said "supported"
        # v6fix8 ① graduation / ③ gap gate / ⑤ enabler budget bookkeeping:
        "consecutive_graduate": 0,   # consecutive snapshots at/above graduate_sr (fast-climb exit)
        "gap_sessions": 0,           # consecutive siege decisions with a big drill-transfer gap
        "gap_forced": False,         # gap gate fired -> required form for this wall is DEPTH
        "siege_sessions": 0,         # siege decisions this focus has consumed (enabler budget)
        "opened_by": opened_by,      # llm | auto (v6fix8 ② ranked_walls auto-open)
        # v6fix10 bookkeeping:
        "gateway_for": None,         # ① set when this focus is the DOOR opened in place of a wall
        "resume_lock": 0,            # ② anti-oscillation: readings left before yield may re-fire
        "attrib_depth_required": False,  # ④ verified access-blocked attribution -> forced DEPTH
        # v7 spawn-anneal relay: None for a normal siege focus; a dict (see _new_relay) while this
        # wall is attacked backward from a deep spawn. "relay_sewn" flips True when the anneal
        # reaches natural spawn — the focus then behaves as a normal held-out-driven siege focus.
        "relay": None,
        "relay_sewn": False,
    }


def _new_relay(r0_floor: int, session_idx: int) -> dict:
    """Fresh relay state for a campaign whose R0 spawns at ``r0_floor`` (LLM-proposed, >= 1).

    ``spawn_floor`` is the operative value: every siege level for this wall must spawn there
    (validator R6_SPAWN); it counts DOWN toward 0 (= natural spawn) on rung graduations and back UP
    (never past r0_floor) on rung regressions. rung index i = r0_floor - spawn_floor."""
    return {
        "r0_floor": int(r0_floor),
        "spawn_floor": int(r0_floor),
        "rung_trained": [],          # fresh trained-SR readings at the CURRENT rung (cleared on transition)
        "rung_graduate_streak": 0,   # consecutive readings >= rung_graduate_sr
        "rung_stall_streak": 0,      # consecutive readings < rung_floor_sr (drives regression)
        "best_rung_trained": None,   # new-high ratchet at the current rung (momentum / early-stop)
        "stall_patience": 0,         # readings with no rung transition and no trained new-high
        "rung_history": [            # append-only transition log (offline inspection + journal)
            {"session": int(session_idx), "event": "relay_opened", "spawn_floor": int(r0_floor)}
        ],
        # v7fix4.6 cliff-split sub-rungs + oscillation liveness (old on-disk relays lack these
        # keys — every reader uses .get with these defaults, so a resumed notebook is fine):
        "sub_stage": 0,              # 0 = FULL (the pre-4.6 whole-floor rung); 1..max = scaffold
                                     # stages (1..5 pre-5.3; 1..8 under rung_descent_regime)
        "sub_stage_by_floor": {},    # str(floor) -> resume stage when re-entering that floor
        "best_by_rung": {},          # "floor:stage" -> best trained ever seen at that rung —
                                     # transitions RESTORE it (a revisit's cheap re-climb is no
                                     # "new high"; patience burns across oscillation, fix9 #2)
        "regress_count": 0,          # regress-family moves so far (cap: relay_max_regressions)
        # v7fix4.7 Q2 DEFEND-driven patience (same on-disk backward-compat rule as above):
        "ratchet_log": [],           # per reading: 1 = strict new absolute max at this rung (ANY size)
        "defends_used": 0,           # defence windows consumed this campaign (cap: relay_defend_budget)
        "defend_pending": None,      # session idx of an OPEN defence window, else None
    }


class SiegeNotebook:
    """Persistent siege journal. Owns the B-layer hard constraints; the LLM (A) only proposes."""

    def __init__(self, path: str | None = None, thresholds: "SiegeThresholds | None" = None):
        self.path = path or _DEFAULT_PATH
        # B-layer thresholds: config-resolved when passed, else the module-constant defaults.
        self.th = thresholds or SiegeThresholds()
        self._nb: dict = self._load()
        # Diagnostics for the caller to log (why a focus was/wasn't set; whether a conquest fired this
        # session). Set fresh each apply_llm_update; None between calls.
        self.last_focus_decision: str | None = None
        self.last_conquest: str | None = None
        # v6fix8 diagnostics: graduation (①), enabler-budget retirement (⑤), auto-open (②).
        self.last_graduation: str | None = None
        self.last_budget_retire: str | None = None
        self.last_auto_open: str | None = None
        # v6fix10 diagnostics: yield/resume (②), door substitution (①).
        self.last_yield: str | None = None
        self.last_resume: str | None = None
        self.last_door_sub: str | None = None
        # v7 diagnostics: relay campaign opened this session.
        self.last_relay_open: str | None = None
        # v6fix10 per-call evidence context (set by apply_llm_update; empty between calls).
        self._call_forensics: dict = {}
        self._call_forensics_provided: bool = False
        self._call_chain_incomplete: set[str] = set()
        # v6fix10.1 hazard-3c: last profile seen by apply_llm_update — render_for_prompt uses it
        # (one session stale, fine for a journal warning) to flag door-locked ACTIVE foci.
        self._last_profile: dict = {}

    # ---- persistence (mirrors StudentProfileLog: atomic write, resume-safe) --------------------

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return _empty_notebook()
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return self._coerce(data)
        except (json.JSONDecodeError, OSError):
            return _empty_notebook()

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._nb, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _coerce(data: object) -> dict:
        """Force a loaded object into the guaranteed schema, filling any missing key with its empty.

        §2.6 backward-compat: a notebook written by the OLD single-focus schema (top-level ``focus``
        string + ``focus_*`` fields + a single ``prereq_tree``) is migrated into the new ``foci`` list
        so old on-disk journals / old unit-test fixtures don't break.
        """
        base = _empty_notebook()
        if not isinstance(data, dict):
            return base
        for key in base:
            if key in data:
                base[key] = data[key]
        # migrate an old single-focus notebook that predates the foci list.
        if not data.get("foci") and data.get("focus"):
            migrated = _empty_focus(
                str(data["focus"]).lower(),
                data.get("focus_started_session"),
                data.get("focus_best_sr"),
            )
            tree = data.get("prereq_tree")
            if isinstance(tree, list):
                migrated["prereq_tree"] = tree
            base["foci"] = [migrated]
        return base

    # ---- read accessors ------------------------------------------------------------------------

    def snapshot(self) -> dict:
        """A deep-ish copy of the current notebook (safe to hand to prompt builders / tests)."""
        return json.loads(json.dumps(self._nb))

    def foci(self) -> list[dict]:
        """All active focus elements (§2.6). Empty list = no active siege."""
        return list(self._nb.get("foci", []))

    def focus_skills(self) -> list[str]:
        """Just the skill names of the active foci, in order (primary first)."""
        return [f["skill"] for f in self._nb.get("foci", []) if isinstance(f.get("skill"), str)]

    @property
    def focus(self) -> str | None:
        """Backward-compat single-focus accessor: the PRIMARY (first) active focus, or None.

        Kept so the many `.focus` read sites (logging, co-occurrence hint) keep working; new code that
        needs all parallel sieges should use ``focus_skills()`` / ``foci()``.
        """
        foci = self._nb.get("foci", [])
        return foci[0]["skill"] if foci else None

    def protected_set(self) -> list[str]:
        # v6fix10 ⑤: the stored set (conquered/graduated walls + links) UNION the high-water
        # registry — "everything the student ever truly had" is guarded, not just siege trophies.
        return sorted(
            set(self._nb.get("protected_set", []))
            | {str(s).lower() for s in (self._nb.get("highwater") or {})}
        )

    def verified_chains(self) -> list[dict]:
        return list(self._nb.get("verified_chains", []))

    # ---- v6fix7 P1a: ladder accessors + external progress/type signals --------------------------

    def retired_registry(self) -> dict:
        """skill -> retirement record (count / last_session / failed_notes / SR snapshots)."""
        return dict(self._nb.get("retired", {}))

    def ladder_level_of(self, skill: str) -> int:
        """Current escalation-ladder level (0-3) of an active focus; 0 if not an active focus."""
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                return int(foc.get("ladder_level", 0))
        return 0

    def required_form(self, skill: str) -> str | None:
        """The level TYPE siege levels for this wall MUST use now (code-enforced by the validator).

        Two independent triggers, strongest evidence first:
          - v6fix8 ③ GAP GATE: the drill-transfer gap fired (drills won in their calm sandbox,
            held-out not following, for gap_force_sessions consecutive siege decisions) -> DEPTH
            (full-pressure levels; no more isolation drills for this wall). Measured overfit beats
            the ladder's freeze heuristic, so this takes precedence.
          - v6fix7 P1a L2 ladder: the whole tree froze -> the OPPOSITE of the form that froze
            (DEPTH<->CONSOLIDATE), once we know which form was being used.
        None otherwise (no constraint).
        """
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() != str(skill).lower():
                continue
            # v7: while a relay is live the rung machine owns this wall's attack form — rung
            # levels pick DEPTH/CONSOLIDATE freely at their spawn floor (the next rung is the
            # pressure test, not a forced form). gap_forced/attrib forcing are suspended for
            # relay walls anyway (see note_transfer_gap / apply_llm_update step 7).
            if self._relay_active(foc):
                return None
            if foc.get("gap_forced"):
                return "DEPTH"
            # v6fix10 ④: a VERIFIED access-blocked attribution short-circuits straight to DEPTH —
            # isolation drills teach a fight the student cannot even reach (fix9: kobold drilled
            # s9-17 at 95-100% trained / 0% held-out while the attribution already named the
            # unreached door). The FIRST siege decision may still drill (diagnostic: establishes
            # the trained-side capability ceiling); from the second on, full pressure only.
            if foc.get("attrib_depth_required") and int(foc.get("siege_sessions", 0)) >= 1:
                return "DEPTH"
            if int(foc.get("ladder_level", 0)) < 2:
                return None
            last = str(foc.get("last_siege_type", "")).upper()
            if last == "CONSOLIDATE":
                return "DEPTH"
            if last == "DEPTH":
                return "CONSOLIDATE"
            return None  # form unknown -> cannot force a flip
        return None

    def note_transfer_gap(
        self, skill: str, trained_pct: float | None, held_pct: float | None,
        session_idx: int | None = None,
    ) -> str | None:
        """v6fix8 ③: feed one siege decision's drill-transfer measurement for an ACTIVE focus.

        Called by gen_manager right after it computes the gap hint (best TRAINED SR among this wall's
        siege-tagged levels vs the wall's held-out SR). Counts consecutive over-gap decisions
        (trained >= gap_trained_min AND trained-held >= gap_min_pp); at gap_force_sessions the form
        for this wall is FORCED to DEPTH (see required_form). Before the force fires, a decision
        where the gap closes / the drill regresses / no reading resets the counter. AFTER it fires
        the force is LATCHED (v6fix9 audit): drills stopping is the expected consequence of the
        forcing, so only TRUE convergence (a fresh >= gap_trained_min reading whose gap has closed)
        lifts it — meanwhile the P3 early-stop tracks held-out against the forced-entry baseline
        and retires the wall as STYLE_REJECTED after gap_stall_patience flat decisions (progress on
        the chain/inventory channel buys patience, same constitution as the P1a ladder). Returns a
        short status string for logging, or None if the skill is not an active focus."""
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() != str(skill).lower():
                continue
            # v7: a live relay wall's trained-vs-held gap is 100pp BY DESIGN through every mid
            # rung (mid-rung levels cannot move held-out at all) — the gap gate would force DEPTH
            # in 3 readings and P3 would then STYLE_REJECT a perfectly healthy campaign. The rung
            # machine (note_rung_reading) owns this wall's form, progress and early stop until it
            # sews; the gap gate + P3 resume at the final rung (v7_design.md §3 risk 2).
            if self._relay_active(foc):
                return "relay(rung machine owns this wall — gap gate suspended until sewn)"
            # v7fix4 P4 sentinel: a SEWN relay wall still reading trained >= graduate-rate while
            # held-out stays ~0 means the training distribution differs from the real one on some
            # axis the ladder never annealed (the v7fix3 shape: trained 97% / held-out 0 / gap
            # 97pp for 3 sessions post-SEWN). With system-built relay levels + the kit-strip rung
            # this state is unreachable BY CONSTRUCTION — so a streak of it is an alarm about an
            # unknown fidelity gap, not a curriculum state worth funding: retire through the
            # normal machinery (cooldown + blacklist + journal teaching), campaign archived.
            if isinstance(foc.get("relay"), dict) and foc.get("relay_sewn"):
                _mismatch = (
                    trained_pct is not None
                    and float(trained_pct) >= self.th.rung_graduate_sr
                    and (held_pct is None or float(held_pct) <= self.th.zero_win_max_sr)
                )
                if _mismatch:
                    foc["sandbox_mismatch"] = int(foc.get("sandbox_mismatch", 0)) + 1
                    if foc["sandbox_mismatch"] >= self.th.sandbox_mismatch_readings:
                        _r = foc.get("relay") or {}
                        _sidx = (
                            session_idx if session_idx is not None
                            else self._nb.get("last_session")
                        )
                        _parked = self._retire_or_park(
                            foc, _sidx, "focus_retired_sandbox_mismatch",
                            relay_r0_floor=_r.get("r0_floor"),
                            trained_at_mismatch=float(trained_pct),
                            held_at_mismatch=(None if held_pct is None else float(held_pct)),
                            mismatch_readings=foc["sandbox_mismatch"],
                        )
                        self._nb["foci"] = [
                            f for f in self._nb.get("foci", []) if f is not foc
                        ]
                        self._save()
                        return (
                            f"SANDBOX_MISMATCH (sewn wall trained {float(trained_pct):.0f}% but "
                            f"held-out ~0 for {self.th.sandbox_mismatch_readings} readings — its "
                            f"training worlds differ from the real distribution on an un-annealed "
                            f"axis; "
                            + ("campaign PARKED to WATCH — frontier-starved, resumes when the "
                               "frontier moves" if _parked else
                               "campaign retired, chain expansion required before any reopen")
                            + ")"
                        )
                    self._save()
                    return (
                        f"sandbox_mismatch_watch({foc['sandbox_mismatch']}/"
                        f"{self.th.sandbox_mismatch_readings}: sewn wall trained "
                        f"{float(trained_pct):.0f}% vs held-out ~0)"
                    )
                if foc.get("sandbox_mismatch"):
                    foc["sandbox_mismatch"] = 0
            # v7fix5.0 P2: ACCESS_CAPPED park — CERTIFIED caps only (cond >= 0.6 + sample guards:
            # execution past the frontier has TRANSFERRED — gnome read reach 18.6% / cond 81%, so
            # its 60pp "gap" was upstream access, not style; it burned a forced series + a
            # wrongful STYLE_REJECTED this way). True style disease is untouched: a wall that is
            # REACHED but loses reads cond LOW -> no certificate -> the normal gate below fires
            # exactly as before. The wall parks to WATCH (privileges withdrawn, chain tracking
            # kept, gap counters frozen so a residual stall count cannot snipe it on resume) and
            # the seat frees for the frontier campaign; _process_watch holds it parked while the
            # cap lasts, so there is no park->resume oscillation.
            _cap = self._access_cap(str(skill).lower())
            if _cap and _cap.get("certified"):
                sl = str(skill).lower()
                _sidx = (
                    session_idx if session_idx is not None else self._nb.get("last_session")
                )
                w = dict(foc)
                w["watch_since"] = _sidx
                w["gap_sessions"] = 0
                w["gap_forced"] = False
                w["gap_stall"] = 0
                w.pop("gap_force_baseline", None)
                w["frozen_sessions"] = 0
                self._nb.setdefault("watch", {})[sl] = w
                self._nb["foci"] = [f for f in self._nb.get("foci", []) if f is not foc]
                self._nb.setdefault("history", []).append(
                    {"session": _sidx, "event": "focus_parked_access_capped",
                     "focus": sl, "frontier": _cap.get("frontier"),
                     "reach_frac": _cap.get("reach_frac"), "cond": _cap.get("cond")}
                )
                self._save()
                return (
                    f"ACCESS_CAPPED({_cap.get('frontier')}: reach "
                    f"{int(round(float(_cap.get('reach_frac', 0.0)) * 100))}%, cond "
                    f"{int(round(float(_cap.get('cond', 0.0)) * 100))}% — execution transferred; "
                    f"wall parked to WATCH, siege the frontier link instead)"
                )
            over = (
                trained_pct is not None and held_pct is not None
                and float(trained_pct) >= self.th.gap_trained_min
                and float(trained_pct) - float(held_pct) >= self.th.gap_min_pp
            )
            if foc.get("gap_forced"):
                # ---- FORCED regime (v6fix9 audit, 2026-07-08) --------------------------------
                # Once DEPTH is forced the isolation drills STOP by design, so the trained
                # reading drops or vanishes as an expected CONSEQUENCE of the forcing — that is
                # not evidence the overfit healed. (With the #2 recency fix, treating it as
                # "drill regressed -> lift the force" would unlatch within ~2 sessions and starve
                # the early-stop of its 3-decision runway: force -> drills stop -> unlatch ->
                # drills resume -> re-force, an oscillation that never retires anything.)
                # The force therefore stays LATCHED until either (a) TRUE convergence — a fresh
                # sandbox reading >= gap_trained_min whose gap to held-out has closed — or
                # (b) the focus exits (graduate / conquer / early-stop below).
                converged = (
                    trained_pct is not None and held_pct is not None
                    and float(trained_pct) >= self.th.gap_trained_min
                    and float(trained_pct) - float(held_pct) < self.th.gap_min_pp
                )
                if converged:
                    foc["gap_sessions"] = 0
                    foc["gap_forced"] = False
                    foc.pop("gap_force_baseline", None)
                    foc.pop("gap_stall", None)
                elif held_pct is not None:
                    # v6fix9 P3 early-stop: forced DEPTH is running — is held-out moving?
                    base = foc.get("gap_force_baseline")
                    if base is None:
                        foc["gap_force_baseline"] = float(held_pct)
                    elif float(held_pct) >= float(base) + self.th.gap_stall_min_gain_pp:
                        # real movement: re-baseline (ratchet) and re-earn full patience.
                        foc["gap_force_baseline"] = float(held_pct)
                        foc["gap_stall"] = 0
                    elif foc.get("chain_frontier_advanced"):
                        # P1a-consistent (v6fix9 audit): the chain/inventory channel reported
                        # measurable progress this session (note_chain_progress) — progress
                        # anywhere in the tree buys patience, same constitution as the ladder.
                        foc["gap_stall"] = 0
                    else:
                        foc["gap_stall"] = int(foc.get("gap_stall", 0)) + 1
                        if foc["gap_stall"] >= self._gap_patience(foc):
                            # STYLE-REJECTED: the sandbox is won (trained >= 90%) and full
                            # pressure was forced, yet held-out refuses to follow — this wall is
                            # not a capability gap drills can fix. Normal retirement path:
                            # cooldown + blacklist + failed-tactic archive, identical to L4/budget.
                            sidx = (session_idx if session_idx is not None
                                    else self._nb.get("last_session"))
                            _parked = self._retire_or_park(
                                foc, sidx, "focus_retired_style_rejected",
                                gap_stall=foc["gap_stall"],
                                gap_force_baseline=foc.get("gap_force_baseline"),
                                held=held_pct, trained=trained_pct,
                            )
                            self._nb["foci"] = [
                                f for f in self._nb.get("foci", []) if f is not foc
                            ]
                            self._save()
                            return ("STYLE_REJECTED->PARKED (frontier-starved)" if _parked
                                    else "STYLE_REJECTED")
            elif over:
                foc["gap_sessions"] = int(foc.get("gap_sessions", 0)) + 1
                if foc["gap_sessions"] >= self.th.gap_force_sessions:
                    foc["gap_forced"] = True
                    # v6fix9 P3: entering forced-DEPTH snapshots the held-out SR — the yardstick
                    # the early-stop measures "did the forcing work?" against.
                    foc["gap_force_baseline"] = float(held_pct)
                    foc["gap_stall"] = 0
                    self._nb.setdefault("history", []).append(
                        {"session": session_idx if session_idx is not None
                         else self._nb.get("last_session"),
                         "event": "gap_gate_forced_depth",
                         "focus": foc.get("skill"), "trained": trained_pct, "held": held_pct}
                    )
            else:
                foc["gap_sessions"] = 0
            self._save()
            if foc.get("gap_forced"):
                status = (
                    f"FORCED_DEPTH (stall {int(foc.get('gap_stall', 0))}/"
                    f"{self._gap_patience(foc)})"
                )
            else:
                status = (
                    f"over_gap {foc['gap_sessions']}/{self.th.gap_force_sessions}" if over else "ok"
                )
            return status
        return None

    def _gap_patience(self, foc: dict) -> int:
        """v6fix9 P3 early-stop patience for one focus — COMBAT walls get DOUBLE.

        Calibrated on job 3691755 (2026-07-08): defeat_gnome_warrior (tier4, gap-forced) climbs
        ~0.5-0.75pp/session (held 4->5->5->7->7) — real progress that only crosses the +3pp
        re-baseline ratchet every ~5-6 decisions, so patience 3 would have retired the run's one
        fix8-exclusive tier4 growth point mid-climb; 6 x 0.5pp/session means the ratchet arrives in
        time for any climber worth keeping. Enablers stay at 3 (armour's real forced series retires
        in 3-5 either way). Same COMBAT boundary as the fix8 ⑤ budget exemption (family_of, no new
        prior)."""
        from auction.craftax_achievements import family_of

        mult = 2 if family_of(str(foc.get("skill", ""))) == "COMBAT" else 1
        return self.th.gap_stall_patience * mult

    def note_siege_level_type(self, skill: str, level_type: str) -> None:
        """Record the TYPE of the latest siege level built for this wall (from <level_meta>).

        This is what makes the L2 forced flip computable — without knowing which form has been
        attacking the wall, "switch form" is unenforceable.
        """
        changed = False
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                lt = str(level_type or "").upper()
                if lt and foc.get("last_siege_type") != lt:
                    foc["last_siege_type"] = lt
                    changed = True
        if changed:
            self._save()

    def note_chain_progress(self, skill: str, advanced: bool = True) -> None:
        """P2 chain-frontier signal: failure episodes are dying DEEPER along this wall's chain.

        For an ACTIVE focus: consumed (and cleared) by the next _update_focus_stall — an advance
        counts as live progress, so the ladder never escalates while the student is measurably
        getting closer (SR may still be 0; this is the tier4 patience signal).
        For a RETIRED wall: recorded on its retirement-registry entry, where _has_new_evidence reads
        it as the blacklist escape hatch (design §P1a: "blacklisted until new chain evidence OR the
        break-link frontier advances"). Cleared on the wall's NEXT retirement (stale by then)."""
        if not advanced:
            return
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower():
                foc["chain_frontier_advanced"] = True
                self._save()
                return
        for rskill, reg in (self._nb.get("retired") or {}).items():
            if str(rskill).lower() == str(skill).lower() and isinstance(reg, dict):
                reg["chain_frontier_advanced"] = True
                self._save()
                return

    def _is_conquered_and_held(self, skill: str, latest_profile: dict[str, float]) -> bool:
        """True when this skill has a VERIFIED conquest entry and its current SR still holds at
        mastered — conquered ground is not a wall; if it later slips, sieging it again is legal."""
        for c in self._nb.get("verified_chains", []):
            if isinstance(c, dict) and str(c.get("target", "")).lower() == str(skill).lower() \
                    and c.get("status") == "verified":
                sr = latest_profile.get(str(skill).lower())
                return sr is not None and sr >= self.th.mastered_sr
        return False

    @staticmethod
    def _blacklist_count(reg: dict) -> int:
        """v7fix4.7 Q3: retirements that ended with the rung micro-ratchet still rising were slow
        true climbs cut by a budget, not failed tactics — they cooldown and archive normally but
        do not stack toward the 2-strikes blacklist."""
        return int(reg.get("count", 0)) - int(reg.get("rising_retirements", 0))

    def _has_new_evidence(self, skill: str, reg: dict, latest_profile: dict[str, float]) -> bool:
        """Blacklist escape hatch: has ANYTHING on this wall's chain moved since retirement?

        True if the wall's own SR, or any link SR snapshotted at retirement, has risen by
        >= focus_improve_pp — or if the P2 break-link frontier advanced (failure episodes dying
        measurably deeper along the chain, delivered via note_chain_progress; SR may still be 0) —
        new evidence earns the wall another attempt."""
        if reg.get("chain_frontier_advanced"):
            return True
        base_sr = reg.get("sr_at_retirement")
        cur = latest_profile.get(str(skill).lower())
        if cur is not None and base_sr is not None and cur >= float(base_sr) + self.th.focus_improve_pp:
            return True
        for lskill, lsr0 in (reg.get("link_sr_at_retirement") or {}).items():
            cur_l = latest_profile.get(str(lskill).lower())
            if cur_l is not None and lsr0 is not None and cur_l >= float(lsr0) + self.th.focus_improve_pp:
                return True
        return False

    def chain_targets(self) -> dict[str, list[str]]:
        """P2: the walls whose failure episodes the ChainOrderLog should break-link-mine, each with
        its ORDERED prereq chain (shallow -> deep): every ACTIVE focus (prereq_tree order) plus every
        RETIRED wall (its chain snapshotted at retirement — kept under watch so a frontier advance
        can unlock the blacklist via note_chain_progress even while nobody is sieging it). Retired
        entries predating the snapshot field fall back to link_sr_at_retirement's key order (built in
        prereq_tree order, dict order preserved)."""
        out: dict[str, list[str]] = {}
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if isinstance(skill, str) and skill:
                out[skill.lower()] = [
                    str(l.get("skill")).lower()
                    for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
                ]
        # v6fix10 ②: WATCH walls keep their chain tracking (the momentum readout the resume/
        # graduate decisions run on) even though every siege privilege is withdrawn.
        for wskill, w in (self._nb.get("watch") or {}).items():
            sl = str(wskill).lower()
            if sl not in out and isinstance(w, dict):
                out[sl] = [
                    str(l.get("skill")).lower()
                    for l in w.get("prereq_tree", []) if isinstance(l.get("skill"), str)
                ]
        for rskill, reg in (self._nb.get("retired") or {}).items():
            sl = str(rskill).lower()
            if sl in out or not isinstance(reg, dict):
                continue
            links = reg.get("links_at_retirement")
            if not links:
                links = list((reg.get("link_sr_at_retirement") or {}).keys())
            out[sl] = [str(l).lower() for l in links]
        # v6fix10.1 hazard-3a: pending-track candidates (parked awaiting admission forensics) are
        # mined too — that IS the point of the waiting room. Entries without links cannot be mined
        # and are skipped (the journal asks the modeler for a prereq_tree).
        for pskill, p in (self._nb.get("pending_track") or {}).items():
            sl = str(pskill).lower()
            if sl not in out and isinstance(p, dict) and p.get("links"):
                out[sl] = [str(l).lower() for l in p["links"]]
        return out

    def prereq_links(self) -> list[dict]:
        """Prereq-tree links across ALL active foci (deduped by skill), each with ``skill``/``role``.

        §2.6: was the single focus's tree; now the union over foci. Dedup keeps the first occurrence
        so a shared enabler (e.g. make_iron_pickaxe) appears once even if several foci list it.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if isinstance(skill, str) and skill.lower() not in seen:
                    seen.add(skill.lower())
                    out.append(link)
        return out

    def unmastered_links(self, latest_profile: dict[str, float]) -> set[str]:
        """Skills across ALL active foci's prereq trees that are NOT yet consolidated (per live SR).

        §2.6: the §3.4 Completed-admission gate wants the UNION over every active siege — any such link
        appearing in a proposer's ``Completed`` list is an illegal compression and must be pulled back
        to ``Relevant``. Computed from LIVE SR (not the stored flag) so it is always current even
        between LLM updates. A set, so a shared enabler across foci is naturally deduped.
        """
        out: set[str] = set()
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if not isinstance(skill, str):
                    continue
                sr = latest_profile.get(skill.lower())
                if mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr) != "CONSOLIDATED":
                    out.add(skill.lower())
        return out

    # ---- B-layer: recompute link flags from live SR (never trust the LLM for this) -------------

    def _refresh_link_flags(self, latest_profile: dict[str, float]) -> None:
        """Overwrite every focus's prereq-tree link ``state``/``sr`` from the current held-out profile."""
        for foc in self._nb.get("foci", []):
            for link in foc.get("prereq_tree", []):
                skill = link.get("skill")
                if not isinstance(skill, str):
                    link["state"] = "UNKNOWN"
                    link["sr"] = None
                    continue
                sr = latest_profile.get(skill.lower())
                link["sr"] = sr
                link["state"] = mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr)

    # ---- B-layer: focus scope + stall bookkeeping ----------------------------------------------

    def _is_valid_focus(self, skill: object, latest_profile: dict[str, float]) -> bool:
        """§3.2 SCOPE hard constraint: a legal focus is a real wall, not an easy/saturated skill.

        A focus must (a) be a known achievement name (str), and (b) NOT already be at/above the
        saturation SR. We do NOT here require it to be combat-only or tier3 — the modeler picks WHICH
        wall from evidence (§3.2); code only forbids wasting the siege on ground already held.

        v7fix1 (= v6fix11 port): (a) is now enforced literally — membership in the 67-achievement
        table. r2 opened the hallucinated ``smelt_iron`` as a focus (held-out None forever, burned
        8/8 enabler decisions); a name outside the table can never be measured, so it is never a
        wall. Applies to the relay path too (this check runs before the relay branch).
        """
        if not isinstance(skill, str) or not skill:
            return False
        from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE

        if skill.lower() not in ACHIEVEMENT_TO_VALUE:
            return False  # hallucinated name — not measurable, never a wall
        sr = latest_profile.get(skill.lower())
        if sr is not None and sr >= self.th.saturated_sr:
            return False  # already mastered ground — not a wall, forbidden as focus
        return True

    def _student_is_mature(self, latest_profile: dict[str, float], num_snapshots: int) -> bool:
        """EARLY-TRAINING guard: is the student past the early ramp so a low SR can mean a WALL?

        Pure SR statistics, no tier table (user 2026-07-04). Immature => low SR is NORMAL_EARLY (B),
        not a wall (A), and NO focus may be set. Mature once there are enough snapshots to read a
        trend AND enough skills already at a decent SR that "still weak everywhere" is ruled out.
        """
        if num_snapshots < self.th.maturity_min_snapshots:
            return False
        n_decent = sum(
            1 for v in latest_profile.values() if v is not None and v >= self.th.maturity_skill_sr
        )
        return n_decent >= self.th.maturity_min_mastered

    @staticmethod
    def _slope(readings: list[float]) -> float:
        """Least-squares slope (pp per session) of a short SR window. <2 points -> 0."""
        pts = [float(r) for r in readings if isinstance(r, (int, float))]
        n = len(pts)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(pts) / n
        denom = sum((i - xm) ** 2 for i in range(n))
        if denom == 0:
            return 0.0
        return sum((i - xm) * (pts[i] - ym) for i in range(n)) / denom

    @staticmethod
    def _notes_similar(a: str, b: str, threshold: float = 0.7) -> bool:
        """Token-Jaccard similarity check for 'is this tactic materially different?' (L3 / reopen)."""
        ta = {w for w in str(a or "").lower().split() if len(w) > 2}
        tb = {w for w in str(b or "").lower().split() if len(w) > 2}
        if not ta or not tb:
            return False  # an empty note is never 'the same tactic'
        return len(ta & tb) / len(ta | tb) >= threshold

    def _ladder_level(self, frozen: int) -> int:
        if frozen >= self.th.ladder_l3:
            return 3
        if frozen >= self.th.ladder_l2:
            return 2
        if frozen >= self.th.ladder_l1:
            return 1
        return 0

    def _update_focus_stall(self, latest_profile: dict[str, float]) -> None:
        """v6fix7 P1a: per-focus stall — FROZEN counter over the WHOLE attack tree (adaptive patience).

        A session is frozen for a focus only if NONE of three progress signals fires:
          (a) focus SR — new best (>= +improve_pp) OR positive windowed slope (> slope_min_pp/session;
              defeats the fix4 ratchet where oscillation under an old best deferred retirement);
          (b) any prereq link SR making a new best — foundations rising = the siege is biting even
              while the wall itself sits at 0% (tier4 patience, user 2026-07-06);
          (c) chain-frontier advance (P2 failure-episode signal) delivered via note_chain_progress().
        frozen_sessions drives the escalation ladder + retirement (the fix4-era stall_sessions
        ratchet decided nothing and was removed in the v7fix3.1 audit; only its best_sr
        high-water side effect survives below).
        Also tracks consecutive_mastered for the conquest gate (#8 fix).
        """
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            sr = latest_profile.get(skill.lower()) if isinstance(skill, str) else None
            prev_best = foc.get("best_sr")
            progress = False

            # v6fix8 ⑤: one siege decision consumed (drives the enabler budget).
            foc["siege_sessions"] = int(foc.get("siege_sessions", 0)) + 1

            # --- best_sr high-water (the only decision-relevant remnant of the old ratchet:
            #     record-delta, retirement snapshots and the P2 zero-win check all read it) ---
            if sr is not None and (prev_best is None or sr >= prev_best + self.th.focus_improve_pp):
                foc["best_sr"] = sr

            # --- (a) focus SR: new best or live windowed slope ---
            if sr is not None:
                hist = [h for h in foc.get("sr_history", []) if isinstance(h, (int, float))]
                hist.append(float(sr))
                foc["sr_history"] = hist[-max(2 * self.th.slope_window, 12):]
                if prev_best is None or sr >= prev_best + self.th.focus_improve_pp:
                    progress = True
                elif self._slope(foc["sr_history"][-self.th.slope_window:]) > self.th.slope_min_pp:
                    progress = True
                # conquest bookkeeping (#8): consecutive snapshots holding at mastered.
                if sr >= self.th.mastered_sr:
                    foc["consecutive_mastered"] = int(foc.get("consecutive_mastered", 0)) + 1
                else:
                    foc["consecutive_mastered"] = 0
                # v6fix8 ① graduation bookkeeping: consecutive snapshots at/above graduate_sr.
                if sr >= self.th.graduate_sr:
                    foc["consecutive_graduate"] = int(foc.get("consecutive_graduate", 0)) + 1
                else:
                    foc["consecutive_graduate"] = 0

            # --- (b) foundation progress: any link making a new best ---
            link_best = foc.setdefault("link_best", {})
            for link in foc.get("prereq_tree", []):
                lskill = link.get("skill")
                if not isinstance(lskill, str):
                    continue
                lsr = latest_profile.get(lskill.lower())
                if lsr is None:
                    continue
                lb = link_best.get(lskill.lower())
                if lb is None:
                    link_best[lskill.lower()] = float(lsr)  # first reading = baseline, not progress
                elif lsr >= lb + self.th.focus_improve_pp:
                    link_best[lskill.lower()] = float(lsr)
                    progress = True

            # --- (c) external chain-frontier advance (consumed once) ---
            if foc.pop("chain_frontier_advanced", False):
                progress = True

            foc["frozen_sessions"] = 0 if progress else int(foc.get("frozen_sessions", 0)) + 1
            foc["ladder_level"] = self._ladder_level(int(foc["frozen_sessions"]))

    def _verify_conquests(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """v6fix7 P1a (#8 fix): a wall is CONQUERED only after holding mastered_sr for
        conquest_consecutive consecutive snapshots. Only then does its verified_chains entry get
        status='verified', its skills enter the protected_set (rehearsal), and the focus retires
        gracefully. A one-session +delta record stays status='progress' and protects nothing —
        fix4 wrote make_iron_pickaxe into verified/protected at 44%, poisoning the tier4 base."""
        from auction.craftax_achievements import family_of

        kept = []
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if not isinstance(skill, str) or int(foc.get("consecutive_mastered", 0)) < self.th.conquest_consecutive:
                kept.append(foc)
                continue
            sl = skill.lower()
            sr = latest_profile.get(sl)
            links = [
                l.get("skill") for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
            ]
            category = "combat_milestone" if family_of(sl) == "COMBAT" else "enabler"
            self._upsert_experience(
                session_idx, sl, links, sr if sr is not None else float(self.th.mastered_sr),
                category, style_note=str(foc.get("style_note", "")), status="verified",
            )
            protect = {sl} | {l.lower() for l in links if l}
            self._nb["protected_set"] = sorted(set(self._nb.get("protected_set", [])) | protect)
            self._nb.setdefault("history", []).append(
                {"session": session_idx, "event": "focus_conquered", "focus": sl,
                 "sr": sr, "held_snapshots": int(foc.get("consecutive_mastered", 0))}
            )
            self.last_conquest = f"{sl} CONQUERED (SR {sr}%, held {foc.get('consecutive_mastered')} snapshots)"
        self._nb["foci"] = kept

    def _retire_or_park(self, foc: dict, session_idx: int, event: str, **extra) -> bool:
        """v7fix5.2 P0: the single retirement choke-point ROUTER. Every path that used to call
        _archive_retirement calls this instead; the caller still removes the focus from ``foci``
        either way (each site already does).

        A wall whose access-cap verdict says FRONTIER-STARVED — a frontier exists (reach < 35%,
        sample guards met) but the cap is NOT certified (cond < 0.6: the chain past the frontier
        cannot be assessed either way, because < 35% of episodes ever see it) — does not die, it
        HIBERNATES: park to WATCH with the FULL focus dict (relay ladder state, attribution,
        style notes) preserved, no retired-registry count, no cooldown, no blacklist strike, no
        failed-tactic archive. It resumes through _process_watch the moment its park frontier
        moves (+focus_improve_pp vs the SR snapshotted here) or the cap dissolves (reach >= 35%).

        Deliberately NOT a new eviction trigger: this routes a death the existing judges
        (patience / DEFEND / budget / gap gate / L4) already decreed — parking is strictly
        gentler than retiring, so no wall dies faster than before (the gap-threshold eviction
        variant was rejected: it would have parked kobold mid-graduation at s143). A certified
        cap never reaches here (fix5.0 P2 parks it earlier, pre-death); reach >= 35% or no
        frontier -> the old retirement path, byte-identical.

        Structural freebies (asserted by fix52 designcheck, not re-implemented): leaving ``foci``
        frees the relay slot (counted as sum(_relay_active(f) for f in foci)) and the DISCOUNT
        extra-seat share (zero_win_walls scans active foci only).

        Returns True when parked, False when retired (callers adjust their log strings)."""
        sl = str(foc.get("skill") or "").lower()
        cap = self._access_cap(sl)
        if not cap or cap.get("certified"):
            self._archive_retirement(foc, session_idx, event, **extra)
            return False
        w = dict(foc)
        w["watch_since"] = session_idx
        # same interface hygiene as every park/yield path: gap state suspends cleanly and the
        # enabler budget freezes (watch entries never pass _update_focus_stall).
        w["gap_sessions"] = 0
        w["gap_forced"] = False
        w["gap_stall"] = 0
        w.pop("gap_force_baseline", None)
        w["frozen_sessions"] = 0
        frontier = str(cap.get("frontier") or "").lower()
        w["park_event"] = str(event)
        w["park_frontier"] = frontier
        _base = (foc.get("link_best") or {}).get(frontier)
        w["park_frontier_sr"] = float(_base) if isinstance(_base, (int, float)) else None
        self._nb.setdefault("watch", {})[sl] = w
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "focus_parked_frontier_starved", "focus": sl,
             "would_be_event": str(event), "frontier": frontier,
             "reach_frac": cap.get("reach_frac"), "cond": cap.get("cond")}
        )
        self.last_park = (
            f"{sl} PARKED to WATCH instead of retiring (frontier-starved: only "
            f"{cap.get('reach_frac', 0) * 100:.0f}% of held-out episodes ever reach "
            f"{frontier}, so siege budget here cannot move held-out until it rises; campaign "
            f"state fully preserved — auto-resumes when {frontier} moves "
            f"+{self.th.focus_improve_pp:.0f}pp; no cooldown, no blacklist strike)"
        )
        return True

    def _archive_retirement(self, foc: dict, session_idx: int, event: str, **extra) -> None:
        """Archive one focus into the ``retired`` registry + history (shared by L4 stall retirement
        and the v6fix8 ⑤ enabler-budget retirement — cooldown / blacklist / new-tactic-on-reopen all
        apply identically to both paths)."""
        skill = str(foc.get("skill") or "")
        reg = self._nb.setdefault("retired", {}).setdefault(
            skill, {"count": 0, "failed_notes": []}
        )
        reg["count"] = int(reg.get("count", 0)) + 1
        # v7fix4.7 Q3: a retirement cut while the rung micro-ratchet was still rising was a slow
        # TRUE climb stopped by a budget, not a failed tactic — it cooldowns and archives like
        # any other, but _blacklist_count subtracts it from the 2-strikes blacklist.
        if extra.get("ratchet_rising"):
            reg["rising_retirements"] = int(reg.get("rising_retirements", 0)) + 1
        reg["last_session"] = session_idx
        # v7fix3.1: remember HOW it retired — the cooldown waives for a relay re-proposal only
        # when the last retirement was an ordinary (natural-spawn) siege, never when the relay
        # campaign itself stalled out (re-opening the same failed ladder must wait the cooldown).
        reg["last_event"] = str(event)
        reg["sr_at_retirement"] = foc.get("best_sr")
        reg["link_sr_at_retirement"] = dict(foc.get("link_best") or {})
        # P2: an ordered chain snapshot (prereq_tree order, shallow -> deep) so break-link
        # mining keeps tracking this wall while it rests, and any pre-retirement frontier
        # flag is stale by definition — the escape hatch must be earned AFTER this point.
        reg["links_at_retirement"] = [
            str(l.get("skill")).lower()
            for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
        ]
        # v7fix4.5 P1: the VERIFIED attribution survives retirement — succession reads it (a relay
        # that stalled on a verified missing enabler hands the trigger queue to that enabler's
        # chain; without this the diagnosis dies with the campaign it diagnosed).
        fa = foc.get("failure_attribution")
        if isinstance(fa, dict) and fa:
            reg["failure_attribution_at_retirement"] = dict(fa)
        reg.pop("chain_frontier_advanced", None)
        note = str(foc.get("style_note", "")).strip()
        if note:
            reg["failed_notes"] = ([*reg.get("failed_notes", []), note])[-3:]
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": event, "focus": foc.get("skill"),
             "best_sr": foc.get("best_sr"), **extra}
        )

    def _retire_stalled_foci(self, session_idx: int) -> None:
        """v6fix7 P1a L4: retirement fires only after ladder_l4 consecutive FROZEN sessions
        (whole-tree no-progress — see _update_focus_stall), never from the legacy ratchet.

        Retirement archives the failed tactic + an SR snapshot of the chain into the ``retired``
        registry, which _reconcile_foci uses for the cooldown / blacklist / "what's different this
        time" checks, and which render_for_prompt shows the modeler (history made READABLE — the
        fix4 bug was retire→same-session-reopen because the modeler never saw its own retirement).
        """
        kept = []
        for foc in self._nb.get("foci", []):
            frozen = int(foc.get("frozen_sessions", 0))
            if frozen >= self.th.ladder_l4:
                self._retire_or_park(
                    foc, session_idx, "focus_retired_stalled", frozen_sessions=frozen
                )
            else:
                kept.append(foc)
        self._nb["foci"] = kept

    def _graduate_fast_climbers(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """v6fix8 ①: a focus holding >= graduate_sr for graduate_consecutive consecutive snapshots
        GRADUATES to maintenance — a wall the siege pushed over the hump this fast was never a real
        wall; the natural curriculum + rehearsal finish it, and the siege slot + all privileges free
        up for a wall that cannot fall any other way. Runs AFTER _verify_conquests (>=70x2 exits as
        CONQUERED, which outranks graduation).

        Maintenance semantics: the wall+links join the protected_set (forgetting-triggered rehearsal
        holds the ground) but the verified_chains entry stays status='progress' — graduation is NOT
        conquest (#8 semantics preserved: only verified conquests may be compressed as tier4 base).
        """
        kept = []
        for foc in self._nb.get("foci", []):
            if int(foc.get("consecutive_graduate", 0)) < self.th.graduate_consecutive:
                kept.append(foc)
                continue
            # graduation body shared with the fix10 WATCH registry (see _graduate_entry).
            self._graduate_entry(foc, session_idx, latest_profile)
        self._nb["foci"] = kept

    def _retire_budget_exhausted(self, session_idx: int) -> None:
        """v6fix8 ⑤: an ENABLER (non-combat) focus that has consumed enabler_max_sessions siege
        decisions retires through the normal retirement machinery (cooldown + blacklist +
        new-tactic-on-reopen). COMBAT walls are exempt — H1 lives there, and a tier4 fight may
        legitimately need 20+ sessions of foundation work (the ladder guards against true freeze)."""
        from auction.craftax_achievements import family_of

        kept = []
        for foc in self._nb.get("foci", []):
            # v7: a live relay is exempt from the enabler budget REGARDLESS of family — a 4-rung
            # anneal needs 16+ decisions and enter_sewers-type walls are non-COMBAT, so the 8-
            # decision budget would kill every deep-descent campaign mid-ladder. The relay's own
            # early stop (note_rung_reading: no rung transition + no trained new-high for
            # relay_stall_patience readings) is its budget; it resumes ⑤ semantics once sewn.
            if self._relay_active(foc):
                kept.append(foc)
                continue
            skill = str(foc.get("skill") or "").lower()
            spent = int(foc.get("siege_sessions", 0))
            if family_of(skill) != "COMBAT" and spent >= self.th.enabler_max_sessions:
                _parked = self._retire_or_park(
                    foc, session_idx, "focus_retired_budget", siege_sessions=spent
                )
                self.last_budget_retire = (
                    f"{skill} enabler budget exhausted ({spent}/{self.th.enabler_max_sessions} siege "
                    "decisions) -> "
                    + ("PARKED to WATCH (frontier-starved: resumes when its frontier moves; no "
                       "cooldown, no blacklist strike)" if _parked else
                       "retired (cooldown applies)")
                    + "; siege slots are for walls that cannot fall to the natural curriculum"
                )
            else:
                kept.append(foc)
        self._nb["foci"] = kept

    def _may_open_new_focus(
        self, latest_profile: dict[str, float], relay: bool = False,
        access_frontier: bool = False,
    ) -> bool:
        """§2.6 expand gate: may a NEW focus be opened this session?

        True iff (a) there is a free slot (fewer than ``max_focus`` active foci) AND (b) either there
        is NO active focus yet (the first wall is always allowed) OR ANY existing focus has reached
        ``focus_expand_sr`` — "one is going well, so there's slack to carry a new one" (user 2026-07-05).

        v7fix5.0 P2b: ``access_frontier=True`` (the candidate IS a link some tracked wall's
        access-cap verdict names as its frontier) waives condition (b), mirroring the relay waiver
        below: the frontier is not an independent new wall — it is the blocked campaign's own
        prerequisite, and the blocking wall itself pins every focus below ``focus_expand_sr`` BY
        CONSTRUCTION (the fix4.5-P2 hostage shape, one level up). Capacity (a) still applies.

        v7fix3 P3: ``relay=True`` (a spawn-anneal campaign is being opened) waives condition (b):
        the expand condition exists to stop siege-fire fragmentation, but a relay is already
        throttled by relay_max + the rung-momentum discount, and v7fix2 showed a 0% ordinary focus
        holding (b) hostage locks the relay out for the whole run. Capacity (a) still applies.
        """
        foci = self._nb.get("foci", [])
        if len(foci) >= self.th.max_focus:
            return False
        if not foci:
            return True  # first wall
        if relay and self.th.relay_expand_exempt:
            return True  # free slot is enough for a relay campaign (relay_max checked downstream)
        if access_frontier:
            return True  # v7fix5.0 P2b: free slot is enough for a named access frontier
        # v7fix4.5 P2 (the mirror of P3 above): an ACTIVE RELAY focus's held-out SR is 0 by
        # construction through every mid rung, so it can never satisfy (b) — during a 30-session
        # campaign it held ALL ordinary expansion hostage (job 3936082 s112:
        # expand_refused(defeat_gnome_warrior) while the modeler was trying to open exactly the
        # combat sub-curriculum its own kobold tactic called for). Test (b) over ORDINARY foci
        # only; if every active focus is a relay, treat as "no ordinary focus yet".
        if self.th.relay_expand_excluded:
            ordinary = [f for f in foci if not self._relay_active(f)]
            if not ordinary:
                return True
            foci = ordinary
        return any(
            (latest_profile.get(f["skill"].lower()) or 0.0) >= self.th.focus_expand_sr
            for f in foci if isinstance(f.get("skill"), str)
        )

    # ---- v6fix10: reachability economics (door gate / yield-to-momentum / high-water) -----------

    def watch_registry(self) -> dict:
        """skill -> parked focus dict (v6fix10 ② WATCH: yielded to natural momentum)."""
        return dict(self._nb.get("watch") or {})

    # ---- v7fix5.0: access-cap verdicts (fed from ChainOrderLog.access_frontier) ------------------

    def note_access_caps(self, caps: dict, session_idx: int | None = None) -> None:
        """v7fix5.0 P1: store this session's access-frontier verdicts (wall -> access dict).

        Called by gen_manager right after it builds the forensics pack — the notebook never
        computes forensics itself, it only consumes the verdicts (same division of labour as
        note_chain_progress). Overwrites wholesale: a cap is a per-session reading, not a ratchet."""
        clean = {
            str(k).lower(): dict(v) for k, v in (caps or {}).items()
            if isinstance(v, dict) and v.get("frontier")
        }
        self._nb["access_caps"] = {"session": session_idx, "caps": clean}
        self._save()

    def _access_cap(self, skill: str) -> dict | None:
        """The stored access-cap verdict for one wall (None when no frontier was found)."""
        entry = self._nb.get("access_caps") or {}
        cap = (entry.get("caps") or {}).get(str(skill).lower())
        return dict(cap) if isinstance(cap, dict) and cap.get("frontier") else None

    def access_frontiers(self) -> set[str]:
        """v7fix5.0 P2b: every link currently named as an access frontier by ANY tracked wall
        (active / watch / retired — whatever gen_manager computed caps for). The expand gate
        exempts opening THESE as new foci: the frontier is not an independent new wall, it is the
        blocked campaign's own prerequisite — refusing it because the blocked campaign itself
        reads 0% is the fix4.5-P2 hostage shape all over again, one level up."""
        entry = self._nb.get("access_caps") or {}
        return {
            str(v.get("frontier")).lower()
            for v in (entry.get("caps") or {}).values()
            if isinstance(v, dict) and v.get("frontier")
        }

    def _access_root_of(self, wall: str) -> str | None:
        """v7fix5.2 P1: chase a capped wall's frontier chain to its ROOT — the link at the end of
        the frontier->frontier chain that has no frontier of its own.

        Why the chase is mandatory (fix51-run forensics, s242): fireball/diamond/kobold/enchant
        were all capped on enter_gnomish_mines, but mines ITSELF was CERTIFIED-capped on
        make_iron_armour — nominating mines would be parked by fix5.0 P2 at the very next
        boundary; only the root (armour: every chain link >= 35%) is attackable. Cycle guard:
        visited-set + a depth bound of the caps table size; a cycle degrades to "stop where we
        are" (still a valid, deeper-than-input target), never an infinite loop.

        A root is by definition NOT starved (starved requires a frontier; the root has none), so
        P0's park routing can never park the root itself — no route loop is possible."""
        caps = (self._nb.get("access_caps") or {}).get("caps") or {}
        cap = caps.get(str(wall).lower())
        if not (isinstance(cap, dict) and cap.get("frontier")):
            return None
        seen = {str(wall).lower()}
        f = str(cap["frontier"]).lower()
        while (
            f not in seen and len(seen) <= len(caps) + 1
            and isinstance(caps.get(f), dict) and caps[f].get("frontier")
        ):
            seen.add(f)
            f = str(caps[f]["frontier"]).lower()
        return f

    def _access_auto_nominate(
        self, session_idx: int, latest_profile: dict[str, float]
    ) -> None:
        """v7fix5.2 P1: deterministic ACCESS-ROOT nomination — the motor fix5.0's C11 lacked.

        fix5.0 bet that rendering BINDING-ACCESS into the chain hint would make the proposer
        nominate the frontier within 1-2 sessions; the fix51 run falsified it (16 sessions, a
        healthy modeler, the instruction ignored 8/8, the one free seat spent on learn_fireball).
        This routine converts the diagnosis the system already computes into a seat allocation:
        chase every capped wall to its root, and if the consensus root is not being worked, open
        it as ``opened_by="access_auto"`` THROUGH EVERY EXISTING ADMISSION GATE (scope /
        conquered / maintenance / cooldown / blacklist incl. its escape hatches / expand with the
        fix5.0 P2b frontier exemption). A gate veto = stand down this boundary, retry next — no
        bypass, no exemption, no notebook surgery. Runs BEFORE _reconcile_foci so the root gets
        first pick of a free seat; the LLM keeps its full freedom on the remaining seats.

        Knowledge-leakage audit: no skill name appears in code; every input is the student's own
        held-out forensics (access_caps <- ChainOrderLog.access_frontier <- fail_hist). From a
        fresh run there are no caps -> this is inert; it engages exactly when a starved frontier
        cluster exists (the regime where the nomination market historically deadlocked).

        Form: an enter_* root opens as an ACCESS-LINK relay (fix5.0 P0.2; R0 via _anchored_r0's
        native-1 semantics) when the relay slot is free; slot busy -> ordinary DEPTH siege (the
        historical mines-campaign form) — never queue-block behind an active relay (N3)."""
        caps = (self._nb.get("access_caps") or {}).get("caps") or {}
        if not caps:
            return
        active = {
            str(f.get("skill", "")).lower() for f in self._nb.get("foci", [])
            if isinstance(f.get("skill"), str)
        }
        watching = {str(s).lower() for s in (self._nb.get("watch") or {})}
        votes: dict[str, int] = {}
        for wall in caps:
            root = self._access_root_of(wall)
            if root:
                votes[root] = votes.get(root, 0) + 1
        candidates = [r for r in votes if r not in active and r not in watching]
        if not candidates:
            return
        # most dependent walls first; tie -> higher current held-out SR (the shallower, more
        # learned link is the cheaper campaign); final tie -> name (deterministic).
        candidates.sort(
            key=lambda r: (-votes[r], -float(latest_profile.get(r) or 0.0), r)
        )
        root = candidates[0]
        veto = self._access_auto_veto(root, session_idx, latest_profile)
        if veto:
            self.last_access_auto = f"access_auto_vetoed({root}: {veto})"
            return
        relay_r0 = None
        if root.startswith("enter_"):
            n_relays = sum(1 for f in self._nb.get("foci", []) if self._relay_active(f))
            if n_relays < self.th.relay_max:
                from auction.craftax_achievements import native_floor_of

                _nf = native_floor_of(root)
                if _nf >= 1:
                    relay_r0, _ = self._anchored_r0(root, _nf)
        self._open_focus(
            root, session_idx, latest_profile, opened_by="access_auto",
            relay_r0_floor=relay_r0,
        )
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "access_auto_opened", "focus": root,
             "dependent_walls": int(votes[root]),
             "form": ("access_link_relay" if relay_r0 is not None else "ordinary")}
        )
        self.last_access_auto = (
            f"access_auto_opened({root}: the access ROOT — {votes[root]} tracked wall(s)' "
            f"frontier chains converge on it"
            + (f"; opened as ACCESS-LINK relay @ R0 spawn_floor={relay_r0}"
               if relay_r0 is not None else "; opened as ordinary DEPTH siege")
            + ")"
        )
        self._save()

    def _access_auto_veto(
        self, sl: str, session_idx: int, latest_profile: dict[str, float]
    ) -> str | None:
        """v7fix5.2 P1: the access_auto candidate walks the SAME admission gates a proposal walks
        (reconcile order), minus the LLM-only new-tactic check (there is no proposal note; the
        ACCESS_CAPPED verdict itself is the categorically-new evidence) and minus the door gate
        (the root chase IS the door logic, computed deeper and deterministically). Returns the
        veto reason, or None when admission is clean."""
        if not self._is_valid_focus(sl, latest_profile):
            return f"scope: SR {latest_profile.get(sl)} >= saturated {self.th.saturated_sr}"
        if self._is_conquered_and_held(sl, latest_profile):
            return "conquered and still held at mastered"
        maint = self._maintenance_block_reason(sl, latest_profile)
        if maint:
            return f"maintenance holds: {maint}"
        reg = (self._nb.get("retired") or {}).get(sl)
        if reg:
            last_ret = int(reg.get("last_session", -10**9))
            cooldown_left = self.th.cooldown_sessions - (session_idx - last_ret)
            if cooldown_left > 0:
                return f"cooldown: {cooldown_left} session(s) left"
            if self._blacklist_count(reg) >= self.th.blacklist_retirements and not \
                    self._has_new_evidence(sl, reg, latest_profile):
                return f"blacklisted: retired {reg.get('count')}x, no new chain evidence"
        if not self._may_open_new_focus(
            latest_profile, relay=False, access_frontier=True,
        ):
            return (
                f"no free focus slot ({len(self._nb.get('foci', []))}/{self.th.max_focus})"
            )
        return None

    def zero_win_walls(self, latest_profile: dict[str, float]) -> set[str]:
        """v6fix10 ③: active foci with NO held-out win in evidence — discount targets.

        v6fix10.1 hazard-2: "no win" = SR never above ``zero_win_max_sr`` (default 1.0pp), not
        strictly zero. A single fluke episode out of 1024 reads 0.098% and must not permanently
        ratchet an effectively-unreached wall to full price. A real first breakthrough (gnome went
        0.3 -> 1.7 -> 2.8) crosses 1.0 within one extra reading, so the unlock is barely delayed."""
        out: set[str] = set()
        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if not isinstance(skill, str):
                continue
            # v7: a relay wall's held-out is ZERO through every mid rung BY CONSTRUCTION, so
            # held-out wins are the wrong evidence axis — the discount reads RUNG PROGRESS
            # instead (v7_design.md §2.2: "rung 在推进=有战果，不贴现；rung 停滞才半价").
            if self._relay_active(foc):
                if self._relay_progressing(foc):
                    continue
                out.add(skill.lower())
                continue
            sr = latest_profile.get(skill.lower())
            best = foc.get("best_sr")
            tol = self.th.zero_win_max_sr
            if (sr is None or float(sr) <= tol) and (best is None or float(best) <= tol):
                out.add(skill.lower())
        return out

    # ---- v7 spawn-anneal relay: rung state machine -----------------------------------------------

    @staticmethod
    def _relay_active(foc: dict) -> bool:
        """True while this focus is a live relay campaign (opened with a rung ladder, not yet sewn)."""
        return isinstance(foc, dict) and isinstance(foc.get("relay"), dict) \
            and not foc.get("relay_sewn")

    def _relay_foc(self, skill: str) -> dict | None:
        for foc in self._nb.get("foci", []):
            if str(foc.get("skill", "")).lower() == str(skill).lower() and self._relay_active(foc):
                return foc
        return None

    def relay_walls(self) -> list[str]:
        """Skills of the active (un-sewn) relay campaigns, in foci order."""
        return [
            str(f.get("skill", "")).lower()
            for f in self._nb.get("foci", []) if self._relay_active(f)
        ]

    def required_spawn_floor(self, skill: str) -> int | None:
        """The spawn floor every siege level for this wall MUST declare now (validator R6_SPAWN).

        None when the wall is not under an active relay (levels then spawn naturally, floor 0 —
        no constraint). After the relay sews this also returns None: natural spawn is floor 0 and
        the meta default covers it."""
        foc = self._relay_foc(skill)
        if foc is None:
            return None
        return int(foc["relay"].get("spawn_floor", 0))

    def relay_kit_stripped(self, skill: str) -> bool:
        """v7fix4 P3: True while this wall's relay is in the final KIT_STRIP rung (natural spawn,
        EMPTY kit == held-out distribution). gen_manager builds that rung's levels with no kit."""
        foc = self._relay_foc(skill)
        return bool(foc and foc["relay"].get("kit_strip"))

    def relay_sub_stage(self, skill: str) -> int:
        """v7fix4.6: the wall's current scaffold sub-stage (0 = FULL whole-floor rung)."""
        foc = self._relay_foc(skill)
        return int(foc["relay"].get("sub_stage", 0)) if foc else 0

    # ---- v7fix5.4: code-calibrated quantile radii ------------------------------------------------
    def floor_radii(self, floor: int) -> list[int] | None:
        """The measured quantile radii for ``floor`` (smallest/easiest first), or None."""
        v = (self._nb.get("floor_d2d_radii") or {}).get(str(int(floor)))
        return [int(x) for x in v] if v else None

    def set_floor_radii(self, floor: int, radii) -> None:
        """Persist the measured quantile radii for ``floor`` (gen_manager calibration; code-owned)."""
        self._nb.setdefault("floor_d2d_radii", {})[str(int(floor))] = [int(x) for x in radii]
        self._save()

    def rung_calibration_needed(self, skill: str) -> int | None:
        """The wall's approach floor whose quantile radii are still unresolved, or None.

        None when the quantile ladder is off, there is no relay, the relay sits on its R0
        floor (R0 keeps the fix5.3 plain radius legs), or the floor is already calibrated."""
        if not (self.th.rung_quantile_ladder and self.th.rung_descent_regime):
            return None
        foc = self._relay_foc(skill)
        if foc is None:
            return None
        r = foc["relay"]
        floor = int(r.get("spawn_floor", 0) or 0)
        if floor < 1 or floor == int(r.get("r0_floor", -1)):
            return None
        return None if self.floor_radii(floor) else floor

    def _ladder_shape(self, r: dict) -> tuple[int, int, list[int], bool]:
        """(max_stage, r0_skip_hi, radii, quantile_on) for THIS relay's current floor.

        quantile_on requires ALL of: both master switches on, NOT the R0 floor, and radii
        RESOLVED for the floor (code-calibrated at relay-build time). Everything else falls
        back to the fix5.3 shape byte-for-byte — relay_scaffold and note_rung_reading must
        derive the table from THIS one helper (two hand-maintained copies would drift)."""
        _regime = bool(self.th.rung_descent_regime)
        floor = int(r.get("spawn_floor", 0) or 0)
        if _regime and self.th.rung_quantile_ladder and floor != int(r.get("r0_floor", -1)):
            q = self.floor_radii(floor)
            if q:
                # stages (4+Q)..5 = quantile radius rungs; 4..1 = the entry descent-leg anneals
                return 4 + len(q), 4, q, True
        static_radii = list(self.th.rung_ladder_radii)
        return (5 if _regime else 2) + len(static_radii), (5 if _regime else 2), static_radii, False

    def relay_scaffold(self, skill: str) -> dict | None:
        """v7fix4.6 P1 (+v7fix5.3): build knobs for the wall's CURRENT scaffold sub-stage, or None
        at FULL / kit-strip / no relay (gen_manager then renders the exact pre-4.6 level).

        {"sub_stage": int, "down_ladder_radius": int | None, "monster_credit": int,
         "uplock": bool, "needs_multiplier": float}
          - down_ladder_radius: spawn within this Manhattan radius of the floor's DOWN ladder;
            None = entry (up-ladder) spawn.
          - monster_credit: pre-credited kills on the spawn floor (the down ladder starts
            unlocked). gen_manager must NOT emit it for floor 0 (monsters_killed[0] inits to 10 —
            writing a lower value would LOCK the open overworld ladder), but floor-0 rungs never
            scaffold anyway (kit-strip is floor 0's only form).
          - uplock (v7fix5.3): remove the rung floor's up-ladder ITEM post-build — no escape leg.
          - needs_multiplier (v7fix5.3): TaskParams.needs_depletion_multiplier for the level
            (1.0 = engine default; the survival-clock anneal axis).
        With rung_descent_regime=False the pre-5.3 6-stage table is rendered verbatim (uplock
        always False, needs always 1.0 — the dict keys are still present but inert).
        All dials are generic backward-chaining scaffolds (world geometry + the game's own
        clear-gate + the game's own survival clocks) — no wall-specific knowledge enters here."""
        foc = self._relay_foc(skill)
        if foc is None:
            return None
        r = foc["relay"]
        stage = int(r.get("sub_stage", 0))
        if stage <= 0 or r.get("kit_strip"):
            return None
        # v7fix5.5 P2: a hypothesis-compiled INSERTED rung — its knobs were measured (paired
        # whatif, delta >= the bar) and frozen at compile time; the stage table is bypassed for
        # exactly this one rung. The distinct stage id keeps the reading filter isolating it.
        # v7fix5.7: the light-anneal leg (stage 49) serves the SAME frozen insert knobs — the
        # graduation handler has already annealed knobs["pre_light"] True -> "ladder" there.
        if (
            stage in (RUNG_INSERT_STAGE, RUNG_INSERT_LIGHT_STAGE)
            and isinstance(r.get("stage_insert"), dict)
        ):
            knobs = dict(r["stage_insert"].get("knobs") or {})
            knobs["sub_stage"] = stage
            return knobs
        return self._stage_knobs(r, stage)

    def _stage_knobs(self, r: dict, stage: int) -> dict:
        """v7fix5.5 P0: the stage->knob table, extracted from relay_scaffold so the journal
        render can also compute the NEXT stage's knobs for the scaffold-facts disclosure
        (current vs next diff). ONE table, two callers — a hand-maintained second copy would
        drift (the _ladder_shape lesson). ``stage <= 0`` = FULL: the unscaffolded rung level
        (entry spawn, no pre-light, no credit, up-ladder present, engine clocks)."""
        if stage <= 0:
            return {
                "sub_stage": 0, "down_ladder_radius": None, "monster_credit": 0,
                "uplock": False, "needs_multiplier": 1.0,
            }
        uplock = False
        needs = 1.0
        max_stage, _skip_hi54, radii, _quant54 = self._ladder_shape(r)
        stage = min(stage, max_stage)
        if _quant54:
            # v7fix5.4 quantile table: stages (4+Q)..5 = the floor's MEASURED quantile radii
            # (easiest first), EVERY radius rung under the regime (uplock + slow clocks — the
            # radius probe's dose-response was measured regime-on, and the rung training
            # distribution must match the entry target); stages 4..1 = fix5.3 entry leg verbatim.
            if stage >= 5:
                radius, uplock, needs = radii[max_stage - stage], True, self.th.rung_needs_slow
            elif stage == 4:
                radius, uplock, needs = None, True, self.th.rung_needs_slow
            elif stage == 3:
                radius, uplock, needs = None, True, self.th.rung_needs_mid
            else:  # stages 2/1: the pre-5.3 entry stages, verbatim
                radius = None
        elif self.th.rung_descent_regime:
            # v7fix5.3 table: 8/7/6 = radius stages, 5 = radius[-1]+lock+slow, 4 = entry+lock+slow,
            # 3 = entry+lock+mid, 2 = entry (old stage 2), 1 = entry+half credit.
            if stage >= 6:
                radius = radii[max_stage - stage]
            elif stage == 5:
                radius, uplock, needs = radii[-1], True, self.th.rung_needs_slow
            elif stage == 4:
                radius, uplock, needs = None, True, self.th.rung_needs_slow
            elif stage == 3:
                radius, uplock, needs = None, True, self.th.rung_needs_mid
            else:  # stages 2/1: the pre-5.3 entry stages, verbatim
                radius = None
        else:
            radius = radii[max_stage - stage] if stage >= 3 else None
        credit = (
            self.th.rung_clear_credit_half if stage == 1 else self.th.rung_clear_credit_full
        )
        # v7fix4.7 Q1: on the TARGET floor the clear-gate is not a rung dial — the R0 down-gate
        # stays LOCKED so the campaign cannot leak below its target floor (the radius/pre-light
        # dials are the whole R0 scaffold). v7fix5.3: R0 never reaches stages 5..1 (the state
        # machine skips them both ways), so uplock/needs stay at their defaults there.
        if int(r.get("spawn_floor", 0)) == int(r.get("r0_floor", -1)):
            credit = 0
        return {
            "sub_stage": stage,
            "down_ladder_radius": (int(radius) if radius is not None else None),
            "monster_credit": int(credit),
            "uplock": bool(uplock),
            "needs_multiplier": float(needs),
        }

    @staticmethod
    def _scaffold_fact_clauses(knobs: dict) -> dict[str, str]:
        """v7fix5.5 P0: one stage-knob dict -> one fact clause per knob, WORLD-RULE wording.

        The journal's scaffold-facts disclosure and its next-stage diff are both rendered
        through THIS function from _stage_knobs output, so the disclosed facts can never
        drift from the table that actually builds the levels (designcheck: the sentences
        are computed, not template constants). pre-light rides the radius knob because that
        is exactly what set_starting_floor does: a radius spawn torch-lights both the spawn
        and the down-ladder 9x9 neighbourhoods; an entry spawn stamps NO light — the floor
        keeps only its own light sources. Facts only — no tactics, no researcher-probe
        numbers (the fix53 knowledge-leak boundary)."""
        _rad = knobs.get("down_ladder_radius")
        # v7fix5.5 P2: an explicit pre_light override (probe variants / inserted rungs) decouples
        # the light stamp from the spawn anchor — the disclosure must follow the ACTUAL build.
        # v7fix5.7: "ladder" = the graded-anneal middle rung (down-ladder stamp only).
        _lit = knobs.get("pre_light")
        if _lit is None:
            _lit = _rad is not None
        _stamp57 = "torch-lit (9x9)"     # ONE literal — the B1.3 single-source rule
        if _lit == "ladder":
            _lit_clause = (
                f"the down ladder is {_stamp57}; your spawn is NOT — dark start, "
                f"lit destination"
            )
        elif _lit:
            _lit_clause = f"your spawn and the down ladder are each {_stamp57}"
        else:
            _lit_clause = "NONE — no torch light is stamped; only the floor's own light sources"
        return {
            "spawn": (
                f"within {int(_rad)} tiles of the down ladder" if _rad is not None
                else "at the floor entry (up-ladder)"
            ),
            "pre-light": _lit_clause,
            "clear-gate pre-credit": f"{int(knobs.get('monster_credit') or 0)}/8 kills",
            "up-ladder": (
                "REMOVED (no retreat upward)" if knobs.get("uplock") else "present"
            ),
            "survival clocks": f"{float(knobs.get('needs_multiplier', 1.0) or 1.0):.1f}x",
        }

    @staticmethod
    def _relay_progressing(foc: dict) -> bool:
        """v7 ③-discount evidence for a relay wall: the campaign counts as 'progressing' when a
        rung transition happened within the last 2 readings OR the last 2 readings' new-high gains
        sum to >= rung_momentum_pp. (Static because it must also run on plain dict snapshots in
        tests; thresholds ride on the stored gains, computed at reading time.)"""
        r = foc.get("relay") or {}
        if int(r.get("readings_since_transition", 0)) <= 2:
            return True
        gains = [g for g in (r.get("gain_log") or [])[-2:] if isinstance(g, (int, float))]
        return bool(gains) and sum(gains) >= float(r.get("momentum_pp", RUNG_MOMENTUM_PP))

    def relay_progressing(self, skill: str) -> bool:
        foc = self._relay_foc(skill)
        return bool(foc) and self._relay_progressing(foc)

    # ---- v7fix3 P6: breadth spawn frontier (deep-spawn BREADTH ecology levels) -------------------

    def breadth_frontier(self) -> int:
        """Deepest floor a non-relay BREADTH level may declare as spawn_floor (R6 bound).

        Self-referential ladder, no skill->floor prior in code: floor 1 is always unlocked; floor
        N+1 unlocks once a BREADTH level spawning AT floor N demonstrates trained SR >=
        ``breadth_frontier_sr``. Monotone (never regresses — a too-deep breadth level just loses
        the normal learnability competition). Persisted with the notebook, resume-safe."""
        try:
            return max(1, int(self._nb.get("breadth_frontier", BREADTH_FRONTIER_START)))
        except (TypeError, ValueError):
            return BREADTH_FRONTIER_START

    def note_breadth_frontier_reading(
        self, floor: object, trained_pct: float | None, session_idx: int | None = None
    ) -> str | None:
        """Feed one trained-SR reading of a deep-spawn BREADTH level; advance the frontier when the
        reading is AT the current frontier floor and clears the bar. Returns the advance message
        (for the [breadth][FRONTIER] log) or None."""
        from auction.craftax_achievements import MAX_DUNGEON_FLOOR

        if trained_pct is None:
            return None
        try:
            floor_i = int(floor)
        except (TypeError, ValueError):
            return None
        cur = self.breadth_frontier()
        if floor_i != cur or float(trained_pct) < self.th.breadth_frontier_sr:
            return None
        if cur >= MAX_DUNGEON_FLOOR:
            return None
        self._nb["breadth_frontier"] = cur + 1
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "breadth_frontier_advanced",
             "floor": cur + 1, "trained": float(trained_pct)}
        )
        self._save()
        return (
            f"floor {cur + 1} unlocked for deep-spawn BREADTH levels (a floor-{cur} breadth level "
            f"reached trained {float(trained_pct):.0f}% >= {self.th.breadth_frontier_sr:.0f}%)"
        )

    def note_rung_reading(
        self, skill: str, trained_pct: float | None, session_idx: int | None = None,
    ) -> str | None:
        """v7: feed one siege decision's FRESH trained-SR reading of the CURRENT rung's levels.

        Called by gen_manager (same call site rhythm as note_transfer_gap, which is suspended for
        relay walls). The reading MUST be current-rung-only and session-stamped fresh (fix9 #2:
        a high-water reading would fake-graduate rungs). Drives the whole state machine:

          - GRADUATE: >= rung_graduate_sr for rung_graduate_consecutive consecutive readings ->
            spawn_floor -= 1 (one floor up toward natural spawn); at floor 0 the campaign is SEWN
            (foc['relay_sewn']=True) and every held-out mechanism resumes for this wall.
          - CLIFF SPLIT (v7fix4.6; v7fix4.7 extends to R0): a fresh FULL rung (not kit-strip)
            whose first rung_cliff_readings readings ALL sit <= rung_cliff_sr is a zero-success
            cliff — no gradient, regressing re-consolidates an already-graduated rung for
            nothing. Split to the easiest scaffold sub-stage instead (spawn near the down ladder
            + pre-light; clear-gate pre-credit below R0, LOCKED at R0) and climb the sub-stages
            back to FULL (R0 skips the clear-gate stages 2/1). Fires BEFORE the stall count.
          - DEFENCE WINDOW (v7fix4.7): at patience exhaustion with the micro-ratchet still
            rising (strict new maxima of any size in >= relay_defend_rising_k of the last 3
            readings) and budget left, retirement is HELD one reading; the modeler must cite
            the actual readings in the focus's style_note (verified numerically) to reset
            patience. Uncited/unverified -> retire; rising retirements don't blacklist.
          - SUB-STAGE GRADUATE/REGRESS (v7fix4.6): scaffold stages graduate on x
            rung_substage_graduate_x readings (>= rung_graduate_sr) one stage harder; a scaffold
            stall (same rung_floor_sr x rung_stall_readings boundary) steps one stage easier.
            Only FULL's graduation is the floor graduation. sub_stage_by_floor remembers each
            floor's resume stage across floor regressions.
          - REGRESS:  < rung_floor_sr for rung_stall_readings consecutive readings -> spawn_floor
            += 1 (back down one rung to re-consolidate; never past r0_floor).
          - REGRESS BUDGET (v7fix4.6): every regress-family move (floor, sub-stage, kit-strip
            restore) counts toward relay_max_regressions; the move over budget retires the
            campaign through the normal machinery instead — an oscillating ladder must hand its
            diagnosis to the succession, not orbit until the wall clock (the fix9 #2 oscillation
            family, closed for real this time: best_by_rung also persists each rung's new-high
            ratchet across transitions, so a revisit's cheap re-climb no longer resets patience).
          - EARLY STOP: relay_stall_patience consecutive readings (COMBAT x2, same boundary as
            _gap_patience) with neither a rung transition nor a trained new-high of
            >= gap_stall_min_gain_pp -> the campaign retires through the normal machinery
            (cooldown + blacklist + failed-tactic archive), relay state archived for reopen review.

        Any rung transition sets chain_frontier_advanced (the ladder's progress signal (c)):
        rung progress IS tree progress, so the escalation ladder never freezes a moving relay.
        Returns a status string for the [siege][RUNG] log, or None when the skill is not an
        active relay focus."""
        foc = self._relay_foc(skill)
        if foc is None:
            return None
        r = foc["relay"]
        sidx = session_idx if session_idx is not None else self._nb.get("last_session")
        if trained_pct is None:
            # No current-rung level trained this decision: no evidence either way — counters hold.
            # (The siege quota + extra seats keep rung levels in rotation; a persistent absence
            # shows up here in the log, not as a silent stall verdict.)
            self._save()
            return (
                f"no_fresh_rung_reading (spawn_floor={r.get('spawn_floor')}, counters hold)"
            )
        reading = float(trained_pct)
        hist = [h for h in (r.get("rung_trained") or []) if isinstance(h, (int, float))]
        hist.append(reading)
        r["rung_trained"] = hist[-12:]
        r["readings_since_transition"] = int(r.get("readings_since_transition", 0)) + 1
        r["momentum_pp"] = self.th.rung_momentum_pp  # ride on the dict for _relay_progressing

        # v7fix4.6: rung identity = floor + sub-stage (the kit-strip exam is its own rung).
        def _rung_key(floor: int, stage: int, strip: bool = False) -> str:
            return f"{int(floor)}:{'strip' if strip else int(stage)}"

        cur_key = _rung_key(
            int(r.get("spawn_floor", 0)), int(r.get("sub_stage", 0)), bool(r.get("kit_strip"))
        )
        # new-high accounting (momentum + early-stop share the ratchet).
        best = r.get("best_rung_trained")
        gain = reading - float(best) if best is not None else reading
        r.setdefault("gain_log", []).append(round(max(0.0, gain), 2))
        r["gain_log"] = r["gain_log"][-6:]
        # v7fix5.7-P2' (§3.2): EVERY judgment below runs on the last-RUNG_WIN window mean of
        # this rung's raw readings. Window not full -> no judgment (counters hold at 0).
        _w_vals = hist[-RUNG_WIN:]
        _w3 = (sum(float(v) for v in _w_vals) / len(_w_vals)) if len(_w_vals) >= RUNG_WIN else None
        # win-mean new-high ratchet, persisted per rung (same oscillation-liveness law as
        # best_by_rung): new high = win3 mean beats this rung's best win3 mean by the margin.
        # The single-read +3pp anchor is GONE (T5) — a lone lucky high no longer sets it.
        bw3 = r.setdefault("best_win3_by_rung", {})
        _prev_w3 = bw3.get(cur_key)
        new_high = _w3 is not None and (
            _prev_w3 is None or _w3 > float(_prev_w3) + RUNG_WIN_NEW_HIGH_PP
        )
        if _w3 is not None and (_prev_w3 is None or _w3 > float(_prev_w3)):
            bw3[cur_key] = round(_w3, 4)
        # v7fix4.7 Q2 micro-ratchet log: kept as raw telemetry + the defence citation source
        # only — the DEFEND rising test itself reads the RAW series now (P2' §3.3).
        r.setdefault("ratchet_log", []).append(
            1 if (best is None or reading > float(best)) else 0
        )
        r["ratchet_log"] = r["ratchet_log"][-6:]
        if best is None or reading > float(best):
            r["best_rung_trained"] = reading
        # v7fix4.6 P2: the ratchet also persists per rung — transitions restore it (see
        # _transition), so re-entering a previously visited rung is never a fake "new high".
        bbr = r.setdefault("best_by_rung", {})
        prev_best = bbr.get(cur_key)
        bbr[cur_key] = reading if prev_best is None else max(float(prev_best), reading)
        # streaks — window-mean judged (P2' §3.2: graduate/stall thresholds unchanged, the
        # INPUT is the win3 mean; consecutive counts keep their old meaning one level up).
        r["rung_graduate_streak"] = (
            int(r.get("rung_graduate_streak", 0)) + 1
            if (_w3 is not None and _w3 >= self.th.rung_graduate_sr) else 0
        )
        r["rung_stall_streak"] = (
            int(r.get("rung_stall_streak", 0)) + 1
            if (_w3 is not None and _w3 < self.th.rung_floor_sr) else 0
        )

        def _transition(event: str, new_floor: int, new_stage: int = 0, strip: bool = False) -> None:
            r["spawn_floor"] = int(new_floor)
            r["sub_stage"] = int(new_stage)
            r.setdefault("rung_history", []).append(
                {"session": sidx, "event": event, "spawn_floor": int(new_floor),
                 "sub_stage": int(new_stage), "reading": reading}
            )
            r["rung_trained"] = []
            r["rung_graduate_streak"] = 0
            r["rung_stall_streak"] = 0
            # v7fix4.6 P2 (liveness): RESTORE the destination rung's persisted best instead of
            # None-resetting — the old reset made every post-transition first reading a "new
            # high", so stall_patience cleared each oscillation cycle and the campaign could
            # never retire (floor3<->floor2 forever). None still means a genuinely new rung.
            r["best_rung_trained"] = (r.get("best_by_rung") or {}).get(
                _rung_key(new_floor, new_stage, strip)
            )
            r["gain_log"] = []
            # v7fix4.7 Q2: the micro-ratchet is per-rung (like gain_log), and a transition IS
            # progress — any open defence window is moot.
            r["ratchet_log"] = []
            r.pop("defend_pending", None)
            r["readings_since_transition"] = 0
            # rung progress IS tree progress: the escalation ladder's signal (c) resets its freeze.
            foc["chain_frontier_advanced"] = True
            self._nb.setdefault("history", []).append(
                {"session": sidx, "event": event, "focus": foc.get("skill"),
                 "spawn_floor": int(new_floor), "sub_stage": int(new_stage)}
            )

        def _regress_budget(what: str) -> str | None:
            """v7fix4.6 P2: count one regress-family move; the move OVER budget retires instead.
            Retiring through _archive_retirement keeps the verified attribution consumable by the
            succession — the whole point of bounding the orbit."""
            n = int(r.get("regress_count", 0)) + 1
            if n > self.th.relay_max_regressions:
                _parked = self._retire_or_park(
                    foc, sidx, "focus_retired_relay_stalled",
                    relay_r0_floor=r.get("r0_floor"),
                    relay_spawn_floor=r.get("spawn_floor"),
                    relay_sub_stage=r.get("sub_stage"),
                    relay_best_rung_trained=r.get("best_rung_trained"),
                    relay_regress_overflow=n,
                )
                self._nb["foci"] = [f for f in self._nb.get("foci", []) if f is not foc]
                self._save()
                return (
                    f"RELAY_RETIRED (regress budget exhausted: this {what} would be regress-family "
                    f"move #{n} > {self.th.relay_max_regressions} — an oscillating ladder hands its "
                    f"diagnosis to the succession instead of orbiting; "
                    + ("campaign PARKED to WATCH with the full ladder state — frontier-starved, "
                       "resumes when the frontier moves" if _parked else
                       "campaign archived, cooldown + blacklist apply")
                    + ")"
                )
            r["regress_count"] = n
            return None

        # v7fix5.3: the descent regime inserts 3 stages (lock+slow @r / lock+slow @entry /
        # lock+mid @entry) between the radius leg and the old entry stages; the R0 skip set
        # widens with it (stages 5..1 are ALL descent-leg anneals there).
        _regime = bool(self.th.rung_descent_regime)
        # v7fix5.4: the ladder shape (stage count / R0 skip set / radii source) comes from the
        # SAME helper relay_scaffold renders from — the two must never drift apart.
        max_stage, _r0_skip_hi, _ladder_radii54, _quant54 = self._ladder_shape(r)

        # ---- GRADUATE (checked first: a strong rung should move before anything else) ----------
        # v7fix4.6 P1: a SCAFFOLD stage graduates on x rung_substage_graduate_x (default 1 — a
        # false advance wastes one stage, nothing downstream is poisoned; the x2 confirmation
        # philosophy stays on FULL below, whose graduation IS the floor graduation).
        stage_now = int(r.get("sub_stage", 0))
        # ---- v7fix5.5 P2: the hypothesis-compiled INSERTED rung moves first --------------------
        # Graduate -> back to the stage it was inserted at (the whole point of the insert);
        # stall -> the insert is REMOVED and the normal regress semantics run from that stage —
        # a wrong hypothesis self-heals through the existing ladder (regress budget charged),
        # no LLM retraction path exists. Neither streak met -> patience/defence run as usual.
        _ins55 = (
            r.get("stage_insert")
            if stage_now in (RUNG_INSERT_STAGE, RUNG_INSERT_LIGHT_STAGE)
            and isinstance(r.get("stage_insert"), dict)
            else None
        )
        if _ins55 is not None:
            _floor_i55 = int(r.get("spawn_floor", 0))
            _ret55 = max(0, min(int(_ins55.get("return_stage", 0) or 0), max_stage))
            if int(r.get("rung_graduate_streak", 0)) >= self.th.rung_substage_graduate_x:
                # v7fix5.7 light-anneal leg: a graduated pre_light=True insert does not fall
                # straight back to the dark return stage (one-shot ~-25pp context cliff);
                # it descends ONE light notch first (spawn stamp removed, down ladder still
                # lit) at a distinct stage id. Same insert dict, same hypothesis — only the
                # light knob anneals; the next graduation takes the normal pop path below.
                _k57 = dict(_ins55.get("knobs") or {})
                if stage_now == RUNG_INSERT_STAGE and _k57.get("pre_light") is True:
                    _k57["pre_light"] = "ladder"
                    _ins55["knobs"] = _k57
                    r["stall_patience"] = 0
                    r.setdefault("sub_stage_by_floor", {})[str(_floor_i55)] = (
                        RUNG_INSERT_LIGHT_STAGE
                    )
                    # fresh ratchet on the semantically-new rung (fix53/54 surgery principle)
                    (r.get("best_by_rung") or {}).pop(
                        f"{_floor_i55}:{RUNG_INSERT_LIGHT_STAGE}", None
                    )
                    (r.get("best_win3_by_rung") or {}).pop(
                        f"{_floor_i55}:{RUNG_INSERT_LIGHT_STAGE}", None
                    )
                    _transition(
                        "rung_insert_light_anneal", _floor_i55, RUNG_INSERT_LIGHT_STAGE
                    )
                    self._save()
                    return (
                        f"RUNG_INSERT_LIGHT_ANNEAL (trained {reading:.0f}% at the pre-lit "
                        f"inserted rung — spawn stamp removed, down ladder stays lit: "
                        f"light-anneal leg at stage {RUNG_INSERT_LIGHT_STAGE}, then back to "
                        f"scaffold stage {_ret55} at spawn_floor={_floor_i55})"
                    )
                r.pop("stage_insert", None)
                self._set_hypothesis_status(
                    _ins55.get("hypothesis_id"), "insert_graduated", session=sidx
                )
                r["stall_patience"] = 0
                r.setdefault("sub_stage_by_floor", {})[str(_floor_i55)] = _ret55
                _transition("rung_insert_graduated", _floor_i55, _ret55)
                self._save()
                return (
                    f"RUNG_INSERT_GRADUATED (trained {reading:.0f}% at the hypothesis-compiled "
                    f"inserted rung ({_ins55.get('axis')} {_ins55.get('direction')}) -> back to "
                    f"scaffold stage {_ret55} at spawn_floor={_floor_i55})"
                )
            if int(r.get("rung_stall_streak", 0)) >= self.th.rung_stall_readings:
                r.pop("stage_insert", None)
                self._set_hypothesis_status(
                    _ins55.get("hypothesis_id"), "insert_stalled", session=sidx
                )
                bail = _regress_budget("stall at the hypothesis-inserted rung")
                if bail:
                    return bail
                _at_r0_i55 = _floor_i55 == int(r.get("r0_floor", _floor_i55))
                if (
                    self.th.rung_cliff_split
                    and _ret55 < max_stage
                    and _floor_i55 >= 1
                    and (not _at_r0_i55 or self.th.rung_r0_scaffold)
                ):
                    _new55 = _ret55 + 1
                    if _at_r0_i55 and 1 <= _new55 <= _r0_skip_hi:
                        _new55 = _r0_skip_hi + 1
                    r.setdefault("sub_stage_by_floor", {})[str(_floor_i55)] = _new55
                    _transition("rung_substage_regressed", _floor_i55, _new55)
                    self._save()
                    return (
                        f"RUNG_INSERT_STALLED (trained < {self.th.rung_floor_sr:.0f}% x"
                        f"{self.th.rung_stall_readings} at the inserted rung — hypothesis "
                        f"removed; normal ladder resumes one stage easier: stage {_new55} at "
                        f"spawn_floor={_floor_i55})"
                    )
                if _floor_i55 < int(r.get("r0_floor", _floor_i55)):
                    r.setdefault("sub_stage_by_floor", {})[str(_floor_i55)] = _ret55
                    _transition(
                        "rung_regressed", _floor_i55 + 1,
                        int((r.get("sub_stage_by_floor") or {}).get(str(_floor_i55 + 1), 0)),
                    )
                    self._save()
                    return (
                        f"RUNG_INSERT_STALLED (inserted rung failed at the ladder's easiest "
                        f"margin — hypothesis removed; spawn back DOWN: floor {_floor_i55 + 1} "
                        f"of {r.get('r0_floor')})"
                    )
                # R0 with nowhere lower: restore the return stage; patience is the only exit.
                r.setdefault("sub_stage_by_floor", {})[str(_floor_i55)] = _ret55
                _transition("rung_insert_removed", _floor_i55, _ret55)
                self._save()
                return (
                    f"RUNG_INSERT_STALLED (inserted rung failed at R0 — hypothesis removed, "
                    f"back to scaffold stage {_ret55}; patience is the only exit here)"
                )
        if stage_now > 0:
            if int(r.get("rung_graduate_streak", 0)) >= self.th.rung_substage_graduate_x:
                floor_now = int(r.get("spawn_floor", 0))
                new_stage = stage_now - 1
                # v7fix4.7 Q1 (+v7fix5.3): stages 1.._r0_skip_hi anneal the DESCENT leg
                # (clear-gate / uplock / needs-clock) — meaningless on the target floor. An R0
                # scaffold jumps from the last radius stage to 0 (FULL) directly.
                _r0_jump = ""
                if floor_now == int(r.get("r0_floor", 0)) and 1 <= new_stage <= _r0_skip_hi:
                    new_stage = 0
                    _r0_jump = (
                        f" (R0 skips the descent-leg stages {_r0_skip_hi}..1 — no descent leg)"
                    )
                r["stall_patience"] = 0
                r.setdefault("sub_stage_by_floor", {})[str(floor_now)] = new_stage
                _transition("rung_substage_graduated", floor_now, new_stage)
                self._save()
                return (
                    f"RUNG_SUBSTAGE_GRADUATED (trained {reading:.0f}% at scaffold stage "
                    f"{stage_now} -> stage {new_stage} at spawn_floor={floor_now}; "
                    f"stage 0 = FULL whole-floor rung{_r0_jump})"
                )
        elif int(r.get("rung_graduate_streak", 0)) >= self.th.rung_graduate_consecutive:
            new_floor = int(r.get("spawn_floor", 0)) - 1
            if new_floor <= 0:
                # v7fix4 P3: the floor rungs alone do not reach held-out semantics — every rung
                # (incl. natural spawn) carried a winner-median kit, so "sewn" still meant
                # "kitted". ONE final rung at natural spawn with an EMPTY kit == the held-out
                # distribution exactly (same worlds, same spawn, same empty inventory); only its
                # graduation SEWs. That upgrades SEWN from a process milestone into a result
                # certificate: a wall that sews here has, by construction, already won kitless
                # natural-spawn worlds at graduate-rate — held-out follows, or the P4 sentinel
                # (sandbox_mismatch) fires on an unknown fidelity gap.
                if self.th.relay_kit_strip and not r.get("kit_strip"):
                    r["stall_patience"] = 0
                    # v7fix4.6: mark floor 1's FULL as mastered for the resume memory, and
                    # restore the kit-strip rung's OWN best (key "0:strip") on re-entry.
                    r.setdefault("sub_stage_by_floor", {})[
                        str(int(r.get("spawn_floor", 0)))
                    ] = 0
                    _transition("rung_kit_strip", 0, 0, strip=True)
                    r["kit_strip"] = True
                    self._save()
                    return (
                        f"KIT_STRIP (floor rungs complete from R0 floor {r.get('r0_floor')} -> "
                        f"final rung: natural spawn with an EMPTY kit == held-out distribution; "
                        f"graduating this rung IS the sewn certificate)"
                    )
                _how = (
                    "kitless natural spawn graduated — trained here IS held-out distribution"
                    if r.get("kit_strip") else "natural spawn reached"
                )
                _transition("relay_sewn", 0)
                foc["relay_sewn"] = True
                r["stall_patience"] = 0
                self._save()
                return (
                    f"SEWN ({_how}: rung ladder complete from R0 floor "
                    f"{r.get('r0_floor')}; held-out machinery — gap gate / P3 / graduation / "
                    f"conquest — resumes for this wall)"
                )
            r["stall_patience"] = 0
            # v7fix4.6: FULL mastered here (memory 0) + resume the destination floor at the
            # scaffold stage it was left on (0 for a fresh floor — cliff split covers a cliff).
            r.setdefault("sub_stage_by_floor", {})[str(int(r.get("spawn_floor", 0)))] = 0
            resume = int((r.get("sub_stage_by_floor") or {}).get(str(new_floor), 0))
            _transition("rung_graduated", new_floor, resume)
            self._save()
            return (
                f"RUNG_GRADUATED (trained {reading:.0f}% held x"
                f"{self.th.rung_graduate_consecutive} -> spawn moves UP: floor {new_floor} of "
                f"{r.get('r0_floor')}"
                + (f", resuming scaffold stage {resume}" if resume else "")
                + ")"
            )

        # ---- CLIFF SPLIT (v7fix4.6 P1; v7fix4.7 Q1 extends it to R0) -----------------------------
        # A fresh FULL rung whose first rung_cliff_readings readings ALL sit at/below
        # rung_cliff_sr is a zero-success cliff: 0% success = zero PPO gradient, and regressing
        # would re-consolidate an already-graduated rung for nothing (the 2026-07-13 descent
        # wall: floor-3 point-blank 73% -> floor-2 spawn 0% flat on BOTH arms). Split to the
        # easiest scaffold stage instead. Deliberately checked BEFORE the stall count (2 readings
        # vs 4) so a cliff never reaches the whole-floor regress path. v7fix4.7 Q1: R0 splits too
        # when its takeoff dice miss (zero-shot 0.28% arbitration, 2026-07-14) — its scaffold is
        # the LIT arena away from the entry (radius + pre-light; no clear-gate, see relay_scaffold).
        # The kit-strip exam still never splits (it IS the SEWN certificate — scaffolding it
        # would dilute the claim).
        rt_here = [x for x in (r.get("rung_trained") or []) if isinstance(x, (int, float))]
        _floor_now = int(r.get("spawn_floor", 0))
        _at_r0 = _floor_now == int(r.get("r0_floor", 0))
        if (
            self.th.rung_cliff_split
            and stage_now == 0
            and not r.get("kit_strip")
            and _floor_now >= 1
            and (not _at_r0 or self.th.rung_r0_scaffold)
            and _floor_now <= int(r.get("r0_floor", 0))
            and len(rt_here) == self.th.rung_cliff_readings
            and all(float(x) <= self.th.rung_cliff_sr for x in rt_here)
        ):
            r.setdefault("sub_stage_by_floor", {})[str(_floor_now)] = max_stage
            _transition("rung_cliff_split", _floor_now, max_stage)
            self._save()
            _what = (
                "a LIT arena away from the entry — no descent leg on the target floor, the "
                "clear-gate stays LOCKED; v7fix4.7"
                if _at_r0 else "spawn beside the down ladder, clear-gate pre-credited"
            )
            return (
                f"RUNG_CLIFF_SPLIT (fresh FULL rung at spawn_floor={_floor_now} read <= "
                f"{self.th.rung_cliff_sr:.0f}% x{self.th.rung_cliff_readings} — zero-success "
                f"cliff, no gradient; splitting to scaffold stage {max_stage} ({_what}) and "
                f"climbing the sub-stages back to FULL)"
            )

        # ---- REGRESS ---------------------------------------------------------------------------
        if int(r.get("rung_stall_streak", 0)) >= self.th.rung_stall_readings:
            cur = int(r.get("spawn_floor", 0))
            # v7fix4 P3: regressing OUT of the kit-strip stage restores the kit at floor 1 (the
            # stage below natural spawn) — the student re-consolidates the geared descent before
            # re-attempting the kitless exam. The flag clears so re-graduating floor 1 re-enters
            # KIT_STRIP (never skips to SEWN).
            if r.get("kit_strip"):
                bail = _regress_budget("kit-strip exam failure")
                if bail:
                    return bail
                r.pop("kit_strip", None)
                _transition(
                    "rung_regressed", 1,
                    int((r.get("sub_stage_by_floor") or {}).get("1", 0)),
                )
                self._save()
                return (
                    f"RUNG_REGRESSED (kitless natural-spawn rung failed trained < "
                    f"{self.th.rung_floor_sr:.0f}% x{self.th.rung_stall_readings} -> back to "
                    f"floor 1 WITH kit to re-consolidate; the kit-strip exam re-runs after "
                    f"re-graduation)"
                )
            # v7fix4.6 P1: a stalled stage first steps ONE SUB-STAGE EASIER (FULL -> stage 1,
            # ..., up to the easiest scaffold) — the within-floor ladder absorbs the difficulty
            # step; only the easiest scaffold stage still regresses a whole floor. v7fix4.7 Q1:
            # R0 sub-stages regress too (skipping the descent-only clear-gate stages 2/1); R0's
            # easiest stage has nowhere lower — patience is the only exit there.
            _rg_at_r0 = cur == int(r.get("r0_floor", cur))
            if (
                self.th.rung_cliff_split
                and stage_now < max_stage
                and cur >= 1
                and (cur < int(r.get("r0_floor", cur))
                     or (_rg_at_r0 and self.th.rung_r0_scaffold))
            ):
                bail = _regress_budget(
                    f"stall at scaffold stage {stage_now}" if stage_now
                    else "stall at the FULL rung"
                )
                if bail:
                    return bail
                new_stage = stage_now + 1
                if _rg_at_r0 and 1 <= new_stage <= _r0_skip_hi:
                    # R0 skips the descent-leg stages: FULL stalls onto the radius ladder.
                    new_stage = _r0_skip_hi + 1
                r.setdefault("sub_stage_by_floor", {})[str(cur)] = new_stage
                _transition("rung_substage_regressed", cur, new_stage)
                self._save()
                return (
                    f"RUNG_SUBSTAGE_REGRESSED (trained < {self.th.rung_floor_sr:.0f}% x"
                    f"{self.th.rung_stall_readings} at stage {stage_now} -> easier scaffold "
                    f"stage {new_stage} at spawn_floor={cur})"
                )
            if cur < int(r.get("r0_floor", cur)):
                bail = _regress_budget(
                    "stall at the easiest scaffold stage" if stage_now else "whole-floor stall"
                )
                if bail:
                    return bail
                r.setdefault("sub_stage_by_floor", {})[str(cur)] = stage_now  # resume point
                _transition(
                    "rung_regressed", cur + 1,
                    int((r.get("sub_stage_by_floor") or {}).get(str(cur + 1), 0)),
                )
                # NOT progress: stall_patience deliberately keeps its count — an oscillating
                # ladder (regress -> cheap re-climb -> regress) must still burn patience, or it
                # never retires (the fix9 #2 oscillation family).
                self._save()
                return (
                    f"RUNG_REGRESSED (trained < {self.th.rung_floor_sr:.0f}% x"
                    f"{self.th.rung_stall_readings} -> spawn back DOWN: floor {cur + 1} of "
                    f"{r.get('r0_floor')} to re-consolidate)"
                )
            # already at R0: nowhere deeper to go — only the early stop below can end this.

        # ---- EARLY STOP (v7fix4.7 Q2: with a facts-verified defence window) ---------------------
        if new_high:
            r["stall_patience"] = 0
            r.pop("defend_pending", None)  # a real new high makes any open window moot
        elif _w3 is None:
            # P2': the window is still filling at this rung — no evidence either way, so
            # patience holds (the same law as the no_fresh beat: evidence moves the machine).
            pass
        else:
            r["stall_patience"] = int(r.get("stall_patience", 0)) + 1
            if r["stall_patience"] >= self._relay_patience(foc):
                rising = self._relay_ratchet_rising(r)
                if r.get("defend_pending") is not None:
                    # the window was open for exactly one decision: verify the citation now.
                    r.pop("defend_pending", None)
                    if self._verify_relay_defence(foc):
                        r["defends_used"] = int(r.get("defends_used", 0)) + 1
                        r["stall_patience"] = 0
                        self._nb.setdefault("history", []).append(
                            {"session": sidx, "event": "relay_patience_defended",
                             "focus": foc.get("skill"), "defends_used": r["defends_used"],
                             "readings": [round(float(x), 2)
                                          for x in (r.get("rung_trained") or [])[-3:]]}
                        )
                        self._save()
                        return (
                            f"RELAY_DEFENDED (the modeler cited the actual readings and the "
                            f"micro-ratchet is rising — patience reset; defence "
                            f"{r['defends_used']}/{self.th.relay_defend_budget}; the readings "
                            f"must KEEP making new maxima to earn another window)"
                        )
                    why = "defence window expired UNCITED"
                elif rising and int(r.get("defends_used", 0)) < self.th.relay_defend_budget:
                    r["defend_pending"] = sidx
                    self._nb.setdefault("history", []).append(
                        {"session": sidx, "event": "relay_defence_window",
                         "focus": foc.get("skill"),
                         "readings": [round(float(x), 2)
                                      for x in (r.get("rung_trained") or [])[-3:]]}
                    )
                    self._save()
                    return (
                        f"RELAY_DEFENCE_WINDOW (patience exhausted but the readings are still "
                        f"making new absolute maxima — retirement HELD one reading; the journal "
                        f"now asks the modeler to DEFEND by citing the recent readings, else "
                        f"the campaign retires)"
                    )
                else:
                    why = (
                        "defence budget spent" if rising
                        else "no rung transition and no trained new-high"
                    )
                _parked = self._retire_or_park(
                    foc, sidx, "focus_retired_relay_stalled",
                    relay_r0_floor=r.get("r0_floor"),
                    relay_spawn_floor=r.get("spawn_floor"),
                    relay_sub_stage=r.get("sub_stage"),
                    relay_best_rung_trained=r.get("best_rung_trained"),
                    stall_patience=r["stall_patience"],
                    ratchet_rising=rising,  # v7fix4.7 Q3: rising retirements don't blacklist
                )
                self._nb["foci"] = [f for f in self._nb.get("foci", []) if f is not foc]
                self._save()
                if _parked:
                    return (
                        f"RELAY_PARKED ({why} for "
                        f"{r['stall_patience']} readings at spawn_floor={r.get('spawn_floor')} — "
                        f"frontier-starved: campaign PARKED to WATCH with the full ladder state; "
                        f"resumes when the frontier moves, no cooldown, no blacklist strike)"
                    )
                return (
                    f"RELAY_RETIRED ({why} for "
                    f"{r['stall_patience']} readings at spawn_floor={r.get('spawn_floor')} — "
                    f"campaign archived; cooldown applies"
                    + ("; blacklist EXEMPT: the micro-ratchet was still rising (a budgeted "
                       "slow climb, not a failed tactic)" if rising else "; blacklist applies")
                    + ")"
                )
        self._save()
        _w3_s = f"{_w3:.1f}" if _w3 is not None else "-"
        return (
            f"rung hold (spawn_floor={r.get('spawn_floor')}/{r.get('r0_floor')}, "
            f"stage={int(r.get('sub_stage', 0))}, trained "
            f"{reading:.0f}% (win3 {_w3_s}), graduate {int(r.get('rung_graduate_streak', 0))}/"
            f"{self.th.rung_substage_graduate_x if int(r.get('sub_stage', 0)) else self.th.rung_graduate_consecutive}, "
            f"stall {int(r.get('rung_stall_streak', 0))}/"
            f"{self.th.rung_stall_readings}, patience {int(r.get('stall_patience', 0))}/"
            f"{self._relay_patience(foc)}, regress {int(r.get('regress_count', 0))}/"
            f"{self.th.relay_max_regressions})"
        )

    def _relay_ratchet_rising(self, r: dict) -> bool:
        """v7fix5.7-P2' (§3.3, the E4 fix): rising = the last-4 RAW readings of the CURRENT
        rung are STRICTLY ascending AND the latest one stands at the rung's persisted best.
        The old input was the consumption-time micro-ratchet log, a SUBSEQUENCE of the
        evidence — E4's slow real climbs looked flat through it. Two guards survive from
        the old law: flat noise cannot sustain four strict ascents in a row, and a cheap
        oscillation re-climb UNDER the restored best_rung_trained never qualifies (the
        fix9 #2 family stays closed). Fewer than 4 readings = not rising (evidence, not
        absence)."""
        xs = [float(x) for x in (r.get("rung_trained") or [])[-4:]
              if isinstance(x, (int, float))]
        if len(xs) < 4:
            return False
        if not all(xs[i] < xs[i + 1] for i in range(3)):
            return False
        best = r.get("best_rung_trained")
        return best is None or xs[-1] >= float(best)

    @staticmethod
    def _verify_relay_defence(foc: dict) -> bool:
        """v7fix4.7 Q2: the defence is valid only if the style_note CITES the actual readings —
        >= 2 of the last 3 rung_trained values must appear numerically in the note (tolerance
        0.15 below 10%, 0.6 above — matching the journal's 1dp/integer rendering). Facts are
        verified, narratives are not (the 2026-07-14 lesson: both the modeler's story and the
        first eval read were wrong; only the reading sequence was true)."""
        r = foc.get("relay") or {}
        readings = [float(x) for x in (r.get("rung_trained") or [])[-3:]
                    if isinstance(x, (int, float))]
        if len(readings) < 2:
            return False
        note = str(foc.get("style_note", ""))
        cited = []
        for m in re.findall(r"\d+(?:\.\d+)?", note):
            try:
                cited.append(float(m))
            except ValueError:
                continue
        if not cited:
            return False
        hits = 0
        for v in readings:
            tol = 0.15 if v < 10.0 else 0.6
            if any(abs(c - v) <= tol for c in cited):
                hits += 1
        return hits >= 2

    def _relay_patience(self, foc: dict) -> int:
        """Early-stop patience for a relay campaign — COMBAT walls get DOUBLE (same boundary and
        same calibration rationale as _gap_patience: slow true climbers must not be mid-climb
        killed; enabler-family relays fail fast)."""
        from auction.craftax_achievements import family_of

        mult = 2 if family_of(str(foc.get("skill", ""))) == "COMBAT" else 1
        return self.th.relay_stall_patience * mult

    # ---- v7fix5.5 PROBE-AS-TOOL: request gate, budget, variant table, report lifecycle -----------

    def _probe_budget_left(self, wall: str, session_idx: int) -> dict:
        """Per-kind budget remaining for this wall in the rolling PROBE_BUDGET_WINDOW."""
        used = {"diagnose": 0, "whatif": 0}
        for e in (self._nb.get("probe_ledger", {}) or {}).get(wall, []) or []:
            try:
                s, k = int(e[0]), str(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if k in used and s > int(session_idx) - PROBE_BUDGET_WINDOW_FAST:
                used[k] += 1
        return {k: max(0, PROBE_BUDGET_PER_KIND - v) for k, v in used.items()}

    @staticmethod
    def _verify_probe_justification(foc: dict, text: str) -> bool:
        """Tier-1 (fix47 _verify_relay_defence's citation handshake, lighter bar): >= 1 number in
        the justification must match one of the last 4 rung_trained readings (facts are verified,
        narratives are not). A relay with no readings yet verifies vacuously — there is nothing
        citable, and the stall trigger cannot have fired either."""
        r = foc.get("relay") or {}
        readings = [float(x) for x in (r.get("rung_trained") or [])[-4:]
                    if isinstance(x, (int, float))]
        if not readings:
            return True
        cited = []
        for m in re.findall(r"\d+(?:\.\d+)?", str(text or "")):
            try:
                cited.append(float(m))
            except ValueError:
                continue
        return any(
            abs(c - v) <= (0.15 if v < 10.0 else 0.6) for v in readings for c in cited
        )

    @staticmethod
    def _compile_probe_filter(filt: object) -> tuple:
        """Whitelist predicate compiler for the diagnose filter. Returns (compiled | None, error |
        None). A compile FAILURE is not a rejection: the caller falls back to uniform random
        sampling and records the error (NO reprompt — the anti-negotiation-loop rule)."""
        if filt is None:
            return None, None
        if not isinstance(filt, dict):
            return None, "filter_not_a_dict"
        field = str(filt.get("field", "")).strip()
        op = str(filt.get("op", "")).strip()
        try:
            val = float(filt.get("value"))
        except (TypeError, ValueError):
            return None, "filter_value_not_numeric"
        if field not in PROBE_SENSORS:
            return None, f"unknown_sensor:{field}"
        if op not in PROBE_FILTER_OPS:
            return None, f"bad_op:{op}"
        lo, hi = PROBE_SENSORS[field]
        if not (float(lo) <= val <= float(hi)):
            return None, f"value_out_of_range:{val:g}"
        return {"field": field, "op": op, "value": val}, None

    def probe_variant_knobs(self, wall: str, axis: str, direction: str) -> tuple:
        """Compile a whatif ask into a ONE-STEP variant of the wall's CURRENT stage knobs.

        The LLM picks {axis, direction} from enums; THIS table picks every number (fix50's 0/8
        numeric-decision record is why). Returns (variant_knobs | None, step_desc | None,
        error | None). The variant dict is stage-knob shaped (relay_scaffold keys, plus an
        optional "pre_light" override consumed by the level builder)."""
        foc = self._relay_foc(wall)
        if foc is None:
            return None, None, "not_an_active_relay_wall"
        r = foc["relay"]
        base = self.relay_scaffold(wall) or self._stage_knobs(r, 0)
        k = dict(base)
        easier = direction == "easier"
        _, _skip, radii, _q = self._ladder_shape(r)
        if axis == "uplock":
            # the fix53 what-if measured the LOCK as the easier condition (21.3% vs 14.1%).
            if bool(k.get("uplock")) == easier:
                return None, None, "no_change_on_axis"
            k["uplock"] = easier
            desc = f"uplock {base.get('uplock')} -> {easier}"
        elif axis == "needs_clock":
            ladder = [1.0, float(self.th.rung_needs_mid), float(self.th.rung_needs_slow)]
            cur = float(k.get("needs_multiplier", 1.0) or 1.0)
            idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - cur))
            nxt = idx + (1 if easier else -1)          # slower clock (smaller mult) = easier
            if not 0 <= nxt < len(ladder) or abs(ladder[nxt] - cur) < 1e-9:
                return None, None, "needs_clock_at_boundary"
            k["needs_multiplier"] = ladder[nxt]
            desc = f"survival clocks {cur:.1f}x -> {ladder[nxt]:.1f}x"
        elif axis == "monster_credit":
            steps = sorted({0, int(self.th.rung_clear_credit_half),
                            int(self.th.rung_clear_credit_full)})
            cur = int(k.get("monster_credit") or 0)
            idx = min(range(len(steps)), key=lambda i: abs(steps[i] - cur))
            nxt = idx + (1 if easier else -1)          # more pre-credit = easier
            if not 0 <= nxt < len(steps) or steps[nxt] == cur:
                return None, None, "monster_credit_at_boundary"
            k["monster_credit"] = steps[nxt]
            desc = f"clear-gate pre-credit {cur}/8 -> {steps[nxt]}/8"
        elif axis == "radius":
            cur = k.get("down_ladder_radius")
            if cur is None:
                return None, None, "radius_axis_needs_a_radius_stage"
            try:
                idx = list(radii).index(int(cur))
            except ValueError:
                return None, None, "current_radius_not_on_ladder"
            nxt = idx + (-1 if easier else 1)          # radii easiest (smallest) first
            if not 0 <= nxt < len(radii):
                return None, None, "radius_at_boundary"
            k["down_ladder_radius"] = int(radii[nxt])
            desc = f"spawn radius {int(cur)} -> {int(radii[nxt])} tiles"
        elif axis == "spawn_anchor":
            cur = k.get("down_ladder_radius")
            if easier:
                if cur is not None:
                    return None, None, "already_anchored_to_the_down_ladder"
                # the fix54 question mechanised: entry vs the LARGEST measured radius — same
                # distance band, anchored (and pre-lit) context.
                k["down_ladder_radius"] = int(radii[-1])
                desc = f"spawn anchor entry -> within {int(radii[-1])} of the down ladder"
            else:
                if cur is None:
                    return None, None, "already_at_the_entry"
                k["down_ladder_radius"] = None
                desc = f"spawn anchor within {int(cur)} of the down ladder -> entry"
        elif axis == "pre_light":
            # v7fix5.7: three-level ladder, darkest first. "ladder" = down-ladder stamp only
            # (dark start, lit destination) — the graded-anneal middle rung. One notch per step.
            _levels = [False, "ladder", True]
            cur = k.get("pre_light")
            if cur is None:
                cur = k.get("down_ladder_radius") is not None   # coupled default
            try:
                idx = _levels.index(cur)
            except ValueError:
                idx = _levels.index(bool(cur))
            nxt = idx + (1 if easier else -1)
            if not 0 <= nxt < len(_levels):
                return None, None, "pre_light_at_boundary"
            k["pre_light"] = _levels[nxt]              # explicit override; builder kwarg
            desc = (
                f"pre-light {cur} -> {_levels[nxt]} (spawn anchor unchanged; "
                f"'ladder' = down-ladder 9x9 stamp only)"
            )
        else:
            return None, None, f"bad_axis:{axis}"
        return k, desc, None

    def _admit_probe_request(self, req: object, session_idx: int) -> None:
        """v7fix5.5: fold the modeler's OPTIONAL probe_request through every code gate.

        Accept -> probe_pending + ledger entry; reject -> receipt only (journal-visible; the
        design's silent-refuse rule: no reprompt, no retry loop). Runs inside apply_llm_update
        AFTER reconcile so the relay foci set is current."""
        self.last_probe_decision = None
        if not isinstance(req, dict) or not req:
            return

        def _reject(reason: str) -> None:
            self._nb["probe_receipt"] = (
                f"s{session_idx}: probe_rejected({wall or '?'}/{kind or '?'}: {reason})"
            )
            self.last_probe_decision = self._nb["probe_receipt"]
            self._save()

        wall = str(req.get("wall", "")).lower().strip()
        kind = str(req.get("kind", "")).strip()
        just = str(req.get("justification", ""))
        if kind not in PROBE_KINDS:
            return _reject(f"bad_kind:{kind}")
        foc = self._relay_foc(wall)
        if foc is None:
            return _reject("not_an_active_relay_wall")
        if self._nb.get("probe_pending"):
            return _reject("a_probe_is_already_pending")
        stalled = not self._relay_progressing(foc)
        tier1 = self._verify_probe_justification(foc, just)
        if not (stalled or tier1):
            return _reject(
                "no_trigger (rung is progressing AND the justification cites no verifiable "
                "recent reading)"
            )
        if self._probe_budget_left(wall, session_idx).get(kind, 0) <= 0:
            return _reject(f"budget_exhausted ({PROBE_BUDGET_PER_KIND}/{kind} per "
                           f"{PROBE_BUDGET_WINDOW} sessions)")
        pending = {
            "wall": wall, "kind": kind, "justification": just[:400],
            "requested_session": int(session_idx), "ckpt_step": None,
            "filter": None, "filter_error": None,
            "axis": None, "direction": None, "variant_knobs": None, "step_desc": None,
        }
        if kind == "diagnose":
            filt, ferr = self._compile_probe_filter(req.get("filter"))
            pending["filter"], pending["filter_error"] = filt, ferr
        else:
            axis = str(req.get("axis", "")).strip()
            direction = str(req.get("direction", "")).strip()
            if axis not in PROBE_AXES:
                return _reject(f"bad_axis:{axis}")
            if direction not in PROBE_DIRECTIONS:
                return _reject(f"bad_direction:{direction}")
            vk, desc, err = self.probe_variant_knobs(wall, axis, direction)
            if err:
                return _reject(err)
            pending.update(axis=axis, direction=direction, variant_knobs=vk, step_desc=desc)
        self._nb["probe_pending"] = pending
        self._nb.setdefault("probe_ledger", {}).setdefault(wall, []).append(
            [int(session_idx), kind]
        )
        extra = (
            f", filter={pending['filter']}"
            + (f" (filter_error={pending['filter_error']} -> uniform random)"
               if pending["filter_error"] else "")
            if kind == "diagnose" else f", step: {pending['step_desc']}"
        )
        self._nb["probe_receipt"] = (
            f"s{session_idx}: probe_accepted({wall}/{kind}{extra}) — runs between sessions; "
            f"report on your next page"
        )
        self.last_probe_decision = self._nb["probe_receipt"]
        self._save()

    def probe_pending(self) -> dict | None:
        """The validated request awaiting main-thread execution (deep copy), or None."""
        p = self._nb.get("probe_pending")
        return json.loads(json.dumps(p)) if isinstance(p, dict) else None

    # ---- v7fix5.6: honest rung readings (measurement decoupled from training) ----------

    def note_rung_eval(self, wall: str, payload: dict) -> None:
        """Main-thread delivery of the between-session zero-shot rung eval (run_rung_eval).
        Stored per wall; gen_manager feeds note_rung_reading from HERE and nowhere else for
        relay walls (fix5.6: the archive trained-SR max was inflated by within-session
        adaptation — entry-stage trained 43% vs zero-shot 24%, probe 2026-07-18)."""
        self._nb.setdefault("rung_eval", {})[str(wall)] = {
            "session": int(payload.get("session", -1)),
            "sr": float(payload.get("sr", 0.0)),
            "spawn_floor": int(payload.get("spawn_floor", -1)),
            "sub_stage": int(payload.get("sub_stage", -1)),
            "n_envs": int(payload.get("n_envs", 0)),
        }
        self._save()

    def rung_eval_for(self, wall: str, session_idx: int) -> dict | None:
        """The stored zero-shot eval, ONLY if fresh (measured in the gap right before this
        session) AND taken under the wall's CURRENT (floor, sub_stage) — a transition after
        measurement voids it (the fix4.6 reading-isolation law applied to the new source).
        Stale or mismatched -> None -> counters hold (the no_fresh beat)."""
        e = (self._nb.get("rung_eval") or {}).get(str(wall))
        if not isinstance(e, dict):
            return None
        if int(e.get("session", -1)) < int(session_idx) - 1:
            return None
        foc = self._relay_foc(wall)
        if foc is None:
            return None
        r = foc["relay"]
        if int(e.get("spawn_floor", -1)) != int(r.get("spawn_floor", -2)):
            return None
        # stage identity derives from relay_scaffold on BOTH sides (run_rung_eval stores it
        # from the same helper) — kit-strip (scaffold None -> 0) and INSERT rungs (stage 50)
        # compare consistently; a hand-rolled second derivation would drift.
        _sc56 = self.relay_scaffold(wall)
        if int(e.get("sub_stage", -1)) != int((_sc56 or {}).get("sub_stage") or 0):
            return None
        return e

    def consume_rung_eval(self, wall: str, session_idx: int) -> tuple:
        """v7fix5.7-P2' T1 (fix56设计 §3.1): consume EVERY zero-shot eval into the rung state
        machine AT DELIVERY TIME (run_dicode Step 4d, once per session). The old consumption
        rode the modeler-decision cadence and sampled every other eval — half the judgment
        evidence was dropped on the floor. Returns (eval_sr | None, status_line | None)."""
        if self._relay_foc(wall) is None:
            return None, None
        e = self.rung_eval_for(wall, session_idx)
        ev = float(e["sr"]) if e else None
        st = self.note_rung_reading(wall, ev, session_idx=session_idx)
        return ev, st

    def deliver_probe_report(self, report: dict, session_idx: int) -> None:
        """Main-thread delivery (run_dicode, after the probe rollout): store the report for the
        wall, clear pending, persist. Same cross-thread write pattern as note_chain_progress —
        the report NEVER touches training state or the rung reading stream."""
        pend = self._nb.get("probe_pending") or {}
        wall = str((report or {}).get("wall") or pend.get("wall") or "").lower()
        if not wall:
            return
        stored = {
            **(report or {}),
            "wall": wall,
            "kind": (report or {}).get("kind") or pend.get("kind"),
            "delivered_session": int(session_idx),
        }
        # v7fix5.5 P2: a Tier-2 verify probe carries its hypothesis id + the ladder context it
        # was scheduled under (the stale-context guard compares against these at verdict time).
        # Delivery stays a pure STORE — the verdict/compile runs in the next session's
        # hypothesis_housekeeping, on the same thread as the rest of the state machine.
        if pend.get("verify_hypothesis_id"):
            stored["verify_of"] = pend["verify_hypothesis_id"]
            stored["verify_floor"] = pend.get("verify_floor")
            stored["verify_sub_stage"] = pend.get("verify_sub_stage")
        self._nb.setdefault("probe_reports", {})[wall] = stored
        self._nb["probe_pending"] = None
        self._save()

    def clear_probe_pending(self, reason: str, session_idx: int) -> None:
        """Executor failure path: drop the pending probe with a journal receipt (the budget entry
        stays spent — a crashing probe must not become a free retry loop)."""
        self._nb["probe_pending"] = None
        self._nb["probe_receipt"] = f"s{session_idx}: probe_failed({reason}) — budget spent"
        self._save()

    # ---- v7fix5.5 P2: hypothesis loop (scientist pass -> Tier-1 -> Tier-2 verify -> compile) ----
    # Categories are inferred, interfaces are pinned: the LLM's whole freedom is the four block
    # fields; every gate, every number and the compile action are code. A probe still only
    # MEASURES — the ONE sanctioned training-state write is the compile, which goes through the
    # rung state machine as an inserted sub-stage the existing graduate/stall/regress/patience
    # machinery fully governs.

    def _hypothesis_entry(self, hyp_id: object) -> dict | None:
        for e in self._nb.get("hypothesis_log") or []:
            if isinstance(e, dict) and e.get("id") == hyp_id:
                return e
        return None

    def _set_hypothesis_status(self, hyp_id: object, status: str, session=None, **extra) -> None:
        e = self._hypothesis_entry(hyp_id)
        if e is None:
            return
        e["status"] = status
        if session is not None:
            e["status_session"] = int(session)
        e.update(extra)

    @staticmethod
    def _report_numbers(rep: object) -> list:
        """Every numeric leaf of a probe report (marginals, snapshots, rates, deltas) — the
        citable set for the scientist's Tier-1 evidence check."""
        out: list = []

        def _walk(v):
            if isinstance(v, bool):
                return
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict):
                for x in v.values():
                    _walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _walk(x)

        _walk(rep)
        return out

    def _verify_hypothesis_evidence(self, foc: dict, rep: object, text: str) -> bool:
        """Tier-1 for the HYPOTHESIS block: >= 1 number in ``evidence`` must match the probe
        report's own numbers OR one of the last 4 rung readings (facts are verified, narratives
        are not — the fix47 citation handshake, same tolerances as the probe-request gate)."""
        r = foc.get("relay") or {}
        citable = self._report_numbers(rep)
        citable += [float(x) for x in (r.get("rung_trained") or [])[-4:]
                    if isinstance(x, (int, float))]
        if not citable:
            return True
        cited = []
        for m in re.findall(r"-?\d+(?:\.\d+)?", str(text or "")):
            try:
                cited.append(float(m))
            except ValueError:
                continue
        return any(
            abs(c - v) <= (0.15 if abs(v) < 10.0 else 0.6) for v in citable for c in cited
        )

    def _hyp_verify_budget_left(self, wall: str, session_idx: int) -> int:
        """Tier-2 verify probes remaining for this wall in the rolling window (own ledger kind
        "verify" — the modeler's diagnose/whatif budget is untouched)."""
        used = 0
        for e in (self._nb.get("probe_ledger", {}) or {}).get(wall, []) or []:
            try:
                s, k = int(e[0]), str(e[1])
            except (TypeError, ValueError, IndexError):
                continue
            if k == "verify" and s > int(session_idx) - PROBE_BUDGET_WINDOW:
                used += 1
        return max(0, HYPOTHESIS_VERIFY_PER_WINDOW - used)

    def hypothesis_scientist_due(self, session_idx: int) -> dict | None:
        """The one fresh, un-theorized, non-verify probe report awaiting a scientist pass
        (deep copy), or None. One scientist shot per report, ever."""
        if not self.th.hypothesis_loop:
            return None
        for wall, rep in sorted((self._nb.get("probe_reports") or {}).items()):
            if not isinstance(rep, dict) or rep.get("hypothesized") or rep.get("verify_of"):
                continue
            if int(session_idx) - int(rep.get("delivered_session") or 0) > PROBE_STALE_SESSIONS:
                continue
            if self._relay_foc(wall) is None:
                continue
            return {"wall": wall, "report": json.loads(json.dumps(rep))}
        return None

    def scientist_context(self, wall: str) -> str:
        """The scientist call's user-prompt context: wall + scaffold facts + recent readings +
        the delivered report, all rendered from the SAME single-source helpers the journal uses
        (facts can never drift between the two calls)."""
        foc = self._relay_foc(wall)
        if foc is None:
            return ""
        r = foc["relay"]
        stage = int(r.get("sub_stage", 0))
        knobs = self.relay_scaffold(wall) or self._stage_knobs(r, stage)
        facts = self._scaffold_fact_clauses(knobs)
        recent = [round(float(x), 1) for x in (r.get("rung_trained") or [])[-4:]
                  if isinstance(x, (int, float))]
        lines = [
            f"WALL: {wall} — a spawn-anneal relay campaign training at spawn_floor "
            f"{int(r.get('spawn_floor', 0))}, scaffold stage {stage} "
            f"(target floor {int(r.get('r0_floor', 0))}).",
            "SCAFFOLD FACTS (world rules at this stage, all code-set): "
            + "; ".join(f"{k}: {v}" for k, v in facts.items()) + ".",
            f"RECENT RUNG TRAINED SR at this stage: {recent or '[]'} "
            f"(readings since last ladder transition: "
            f"{int(r.get('readings_since_transition', 0))}).",
        ]
        rep = (self._nb.get("probe_reports") or {}).get(wall)
        if isinstance(rep, dict):
            lines.extend(ln.strip() for ln in self._render_probe_report(rep))
        return "\n".join(lines)

    def admit_hypothesis(self, wall: str, raw: object, session_idx: int) -> None:
        """Fold ONE scientist-pass output through every code gate; record it whatever happens.

        Shape gate (enums) -> Tier-1 (evidence must cite report/reading numbers) -> verdict:
        the triggering report already measured this exact intervention -> immediate verdict
        (free); else a Tier-2 verify probe is scheduled when the rung is stalled, the verify
        budget has room and the probe slot is free — otherwise the entry stays ``recorded`` and
        housekeeping retries within the freshness window. No reprompt on any rejection."""
        self.last_hypothesis_decision = None
        if not self.th.hypothesis_loop:
            return
        wall = str(wall or "").lower().strip()
        rep = (self._nb.get("probe_reports") or {}).get(wall)
        if isinstance(rep, dict):
            rep["hypothesized"] = True   # one shot per report, verdict-independent
        hid = f"h{int(session_idx)}_{wall}_{len(self._nb.get('hypothesis_log') or [])}"

        def _done(txt: str) -> None:
            self.last_hypothesis_decision = f"s{session_idx}: {txt}"
            self._save()

        foc = self._relay_foc(wall)
        if foc is None:
            return _done(f"hypothesis_dropped({wall}: not an active relay wall)")
        raw = raw if isinstance(raw, dict) else {}
        iv = raw.get("intervention") if isinstance(raw.get("intervention"), dict) else {}
        axis = str(iv.get("axis", "")).strip()
        direction = str(iv.get("direction", "")).strip()
        entry = {
            "id": hid, "wall": wall, "session": int(session_idx),
            "hypothesis": str(raw.get("hypothesis", ""))[:600],
            "evidence": str(raw.get("evidence", ""))[:400],
            "axis": axis, "direction": direction,
            "prediction": str(raw.get("prediction", ""))[:300],
            "status": "recorded",
        }
        self._nb.setdefault("hypothesis_log", []).append(entry)
        if not entry["hypothesis"] or axis not in PROBE_AXES \
                or direction not in PROBE_DIRECTIONS:
            entry["status"] = "rejected_shape"
            return _done(
                f"hypothesis_rejected({wall}: bad shape — axis={axis or '?'} "
                f"direction={direction or '?'})"
            )
        if not self._verify_hypothesis_evidence(foc, rep, entry["evidence"]):
            entry["status"] = "rejected_tier1"
            return _done(
                f"hypothesis_rejected({wall}/{axis}->{direction}: evidence cites no "
                f"verifiable report/reading number)"
            )
        # Free verdict: the triggering report IS a paired measurement of this intervention at
        # this very stage — no second probe, no budget.
        r = foc["relay"]
        if (
            isinstance(rep, dict) and rep.get("kind") == "whatif"
            and str(rep.get("axis")) == axis and str(rep.get("direction")) == direction
            and rep.get("sub_stage") is not None
            and int(rep.get("sub_stage")) == int(r.get("sub_stage", 0))
            and isinstance(rep.get("delta_pp"), (int, float))
            and int(session_idx) - int(rep.get("delivered_session") or 0)
            <= PROBE_STALE_SESSIONS
        ):
            delta = float(rep["delta_pp"])
            if delta >= self.th.hypothesis_verify_delta_pp:
                ok = self._try_compile_hypothesis(foc, entry, session_idx, delta)
                return _done(
                    f"hypothesis_{entry['status']}({wall}/{axis}->{direction}: measured "
                    f"delta {delta:+.1f}pp from the triggering whatif"
                    + ("; inserted rung active" if ok else "")
                    + ")"
                )
            entry["status"] = "refuted"
            entry["delta_pp"] = delta
            return _done(
                f"hypothesis_refuted({wall}/{axis}->{direction}: measured delta "
                f"{delta:+.1f}pp < {self.th.hypothesis_verify_delta_pp:.0f}pp, from the "
                f"triggering whatif — no probe spent)"
            )
        if not self._relay_progressing(foc) \
                and self._schedule_hypothesis_verify(foc, entry, session_idx):
            return _done(
                f"hypothesis_verify_scheduled({wall}/{axis}->{direction}: paired one-knob "
                f"measurement runs between sessions)"
            )
        return _done(
            f"hypothesis_recorded({wall}/{axis}->{direction}: verification waits — "
            f"rung progressing, probe slot busy, or verify budget spent)"
        )

    def _schedule_hypothesis_verify(self, foc: dict, entry: dict, session_idx: int) -> bool:
        """Build the Tier-2 verify probe (whatif-shaped pending riding the batch-2 executor
        UNCHANGED) — only when the single probe slot is free and the verify budget has room."""
        wall = str(entry.get("wall", ""))
        if self._nb.get("probe_pending"):
            return False
        if self._hyp_verify_budget_left(wall, session_idx) <= 0:
            return False
        r = foc["relay"]
        if r.get("kit_strip"):
            return False
        # single-insert invariant: while a compiled rung is active, a second verify would only
        # produce an un-compilable verdict (see _try_compile_hypothesis) — don't burn the
        # budget; the entry stays `recorded` and housekeeping reschedules after the insert
        # resolves (graduate/stall), within the freshness window.
        if isinstance(r.get("stage_insert"), dict):
            return False
        vk, desc, err = self.probe_variant_knobs(wall, entry["axis"], entry["direction"])
        if err:
            entry["status"] = "unverifiable"
            entry["note"] = err
            return False
        self._nb["probe_pending"] = {
            "wall": wall, "kind": "whatif",
            "justification": f"tier-2 verify of {entry['id']}",
            "requested_session": int(session_idx), "ckpt_step": None,
            "filter": None, "filter_error": None,
            "axis": entry["axis"], "direction": entry["direction"],
            "variant_knobs": vk, "step_desc": desc,
            "verify_hypothesis_id": entry["id"],
            "verify_floor": int(r.get("spawn_floor", 0)),
            "verify_sub_stage": int(r.get("sub_stage", 0)),
        }
        self._nb.setdefault("probe_ledger", {}).setdefault(wall, []).append(
            [int(session_idx), "verify"]
        )
        entry["status"] = "verify_scheduled"
        return True

    def _try_compile_hypothesis(
        self, foc: dict, entry: dict, session_idx: int, delta: float
    ) -> bool:
        """The ONE sanctioned compile: a verified intervention becomes an INSERTED rung.

        Knobs are recomputed from the CURRENT stage through probe_variant_knobs (single source
        with the measurement), the fix47 Q1 R0 pin is re-applied (the hypothesis loop must not
        open the target floor's down-gate), and the transition bookkeeping mirrors
        note_rung_reading's _transition — best_by_rung / patience / defence all behave as for
        any other rung. On any refusal the verdict is still recorded; nothing compiles."""
        wall = str(entry.get("wall", ""))
        r = foc["relay"]
        if r.get("kit_strip"):
            entry.update(status="compile_refused", note="kit_strip_exam", delta_pp=float(delta))
            return False
        # single-insert invariant: compiling over an ACTIVE insert would overwrite its
        # return_stage with the insert id itself — graduation would then clamp to the easiest
        # ladder stage and the original resume point would be lost. One insert at a time.
        if isinstance(r.get("stage_insert"), dict):
            entry.update(
                status="compile_refused", note="insert_already_active", delta_pp=float(delta)
            )
            return False
        vk, desc, err = self.probe_variant_knobs(wall, entry["axis"], entry["direction"])
        if err:
            entry.update(status="compile_refused", note=err, delta_pp=float(delta))
            return False
        floor = int(r.get("spawn_floor", 0))
        if floor == int(r.get("r0_floor", -1)):
            vk["monster_credit"] = 0   # fix4.7 Q1: the R0 down-gate stays LOCKED, always
        cur = self._stage_knobs(r, int(r.get("sub_stage", 0)))
        if "pre_light" not in vk and all(
            vk.get(k) == cur.get(k)
            for k in ("down_ladder_radius", "monster_credit", "uplock", "needs_multiplier")
        ):
            entry.update(
                status="compile_refused", note="r0_pin_left_no_change", delta_pp=float(delta)
            )
            return False
        knobs = {
            k: vk[k]
            for k in ("down_ladder_radius", "monster_credit", "uplock", "needs_multiplier",
                      "pre_light")
            if k in vk
        }
        k_ret = int(r.get("sub_stage", 0))
        r["stage_insert"] = {
            "return_stage": k_ret, "floor": floor, "knobs": knobs,
            "axis": entry["axis"], "direction": entry["direction"],
            "hypothesis_id": entry["id"], "session": int(session_idx),
            "delta_pp": float(delta), "step_desc": desc,
        }
        r["sub_stage"] = RUNG_INSERT_STAGE
        # transition bookkeeping (mirrors note_rung_reading._transition, which is a closure
        # there): fresh streaks, per-rung ratchet restore, open defence window moot, and the
        # move counts as ladder progress for the escalation freeze.
        r["rung_trained"] = []
        r["rung_graduate_streak"] = 0
        r["rung_stall_streak"] = 0
        # every insert is a NEW task (its own knob combination) — a previous insert's ratchet
        # under the same "floor:50" key would be a stale best on a semantically different rung
        # (the fix53/54 surgery principle). Fresh ratchet, key dropped.
        (r.get("best_by_rung") or {}).pop(f"{floor}:{RUNG_INSERT_STAGE}", None)
        (r.get("best_win3_by_rung") or {}).pop(f"{floor}:{RUNG_INSERT_STAGE}", None)  # P2'
        r["best_rung_trained"] = None
        r["gain_log"] = []
        r["ratchet_log"] = []
        r.pop("defend_pending", None)
        r["readings_since_transition"] = 0
        foc["chain_frontier_advanced"] = True
        r.setdefault("rung_history", []).append(
            {"session": int(session_idx), "event": "hypothesis_insert",
             "spawn_floor": floor, "sub_stage": RUNG_INSERT_STAGE,
             "axis": entry["axis"], "direction": entry["direction"],
             "delta_pp": float(delta), "hypothesis_id": entry["id"]}
        )
        self._nb.setdefault("history", []).append(
            {"session": int(session_idx), "event": "hypothesis_insert",
             "focus": foc.get("skill"), "spawn_floor": floor,
             "sub_stage": RUNG_INSERT_STAGE}
        )
        entry.update(
            status="verified_compiled", delta_pp=float(delta),
            compiled_session=int(session_idx), return_stage=k_ret,
        )
        return True

    def hypothesis_housekeeping(self, session_idx: int) -> None:
        """Once per session, BEFORE the scientist/modeler calls: (1) verdicts for delivered
        Tier-2 verify reports (>= bar -> compile, else REFUTED — fed back to the journal);
        (2) retry verify scheduling for ``recorded`` entries within the freshness window;
        (3) expire the rest. All decisions land in last_hypothesis_decision for the log."""
        self.last_hypothesis_decision = None
        if not self.th.hypothesis_loop:
            return
        notes: list = []
        bar = self.th.hypothesis_verify_delta_pp
        for wall, rep in sorted((self._nb.get("probe_reports") or {}).items()):
            if not isinstance(rep, dict) or not rep.get("verify_of") \
                    or rep.get("verdict_done"):
                continue
            rep["verdict_done"] = True
            entry = self._hypothesis_entry(rep["verify_of"])
            if entry is None:
                continue
            delta = rep.get("delta_pp")
            foc = self._relay_foc(wall)
            if not isinstance(delta, (int, float)):
                entry["status"] = "stale_context"
                entry["note"] = "verify_report_missing_delta"
                notes.append(f"verdict({wall}): malformed verify report — recorded only")
            elif float(delta) < bar:
                entry["status"] = "refuted"
                entry["delta_pp"] = float(delta)
                notes.append(
                    f"verdict({wall}/{entry.get('axis')}->{entry.get('direction')}): "
                    f"REFUTED (delta {float(delta):+.1f}pp < {bar:.0f}pp)"
                )
            elif (
                foc is None
                or int(foc["relay"].get("spawn_floor", -1)) != int(rep.get("verify_floor", -2))
                or int(foc["relay"].get("sub_stage", -1)) != int(rep.get("verify_sub_stage", -2))
            ):
                entry["status"] = "stale_context"
                entry["delta_pp"] = float(delta)
                notes.append(
                    f"verdict({wall}): VERIFIED delta {float(delta):+.1f}pp but the ladder "
                    f"moved — recorded only, nothing compiled"
                )
            else:
                ok = self._try_compile_hypothesis(foc, entry, session_idx, float(delta))
                notes.append(
                    f"verdict({wall}/{entry.get('axis')}->{entry.get('direction')}): "
                    f"VERIFIED (delta {float(delta):+.1f}pp >= {bar:.0f}pp)"
                    + (" — inserted rung compiled into the ladder" if ok
                       else f" — compile refused ({entry.get('note')})")
                )
        # An orphaned verify_scheduled (its pending was dropped by the executor and no report
        # ever delivered) reverts to `recorded` so the retry/expiry machinery below owns it.
        _pend_hid = (self._nb.get("probe_pending") or {}).get("verify_hypothesis_id")
        _rep_hids = {
            rep.get("verify_of")
            for rep in (self._nb.get("probe_reports") or {}).values()
            if isinstance(rep, dict)
        }
        for entry in self._nb.get("hypothesis_log") or []:
            if (
                isinstance(entry, dict) and entry.get("status") == "verify_scheduled"
                and entry.get("id") != _pend_hid and entry.get("id") not in _rep_hids
            ):
                entry["status"] = "recorded"
        for entry in self._nb.get("hypothesis_log") or []:
            if not isinstance(entry, dict) or entry.get("status") != "recorded":
                continue
            if int(session_idx) - int(entry.get("session", 0)) > PROBE_STALE_SESSIONS:
                entry["status"] = "expired"
                notes.append(f"expired({entry.get('wall')}: unverified past the freshness window)")
                continue
            foc = self._relay_foc(str(entry.get("wall", "")))
            if foc is None:
                entry["status"] = "expired"
                notes.append(f"expired({entry.get('wall')}: wall left the relay set)")
                continue
            if not self._relay_progressing(foc) \
                    and self._schedule_hypothesis_verify(foc, entry, session_idx):
                notes.append(
                    f"verify_scheduled({entry.get('wall')}/{entry.get('axis')}->"
                    f"{entry.get('direction')}: slot freed — paired measurement runs "
                    f"between sessions)"
                )
        if notes:
            self.last_hypothesis_decision = f"s{session_idx}: " + "; ".join(notes)
        self._save()

    def _door_substitute(self, sl: str, latest_profile: dict[str, float]) -> str | None:
        """v6fix10 ① DOOR GATE: the first CLOSED link (< door_min_sr held-out) among the wall's
        failures' top-3 missing links — the door to attack INSTEAD of the wall. None = wall is
        reachable (no forensic sample, no missing link, or every substantial missing link is
        already open). The door comes from the student's OWN failure histogram, never a
        course-chain prior. An unknown SR counts as closed (never measured = no evidence it opens).

        v6fix10.1 hazard-3b: scan the top-3, not just rank 1 — a closed door hiding at rank 2/3
        behind an open rank-1 link (kobold s32: sword 39% share at SR 57 over dungeon/sewers) used
        to pass the gate and fall through to ④'s forced-DEPTH with no substitution. Ranks past the
        first need DOOR_MIN_SHARE of failures (noise guard); rank 1 keeps no floor (it is the modal
        missing link by construction).

        v7fix1 (= v6fix11 port): a candidate outside the 67-achievement table is skipped at ANY
        rank — the rank-0 "unknown SR = closed" semantics let r2 open the hallucinated
        ``smelt_iron`` as a door; the scan simply moves to the next real link."""
        from auction.craftax_achievements import ACHIEVEMENT_TO_VALUE

        fx = self._call_forensics.get(sl)
        if not fx:
            return None
        for rank, item in enumerate((fx.get("missing_top") or [])[:3]):
            try:
                door, share = str(item[0]).lower(), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if door == sl:
                continue
            if door not in ACHIEVEMENT_TO_VALUE:
                continue  # v7fix1: hallucinated link name — never a door
            if self.th.tier4_relay_only:
                # v7fix3.1 audit fix: a deepest-layer door (enter_fire_realm & co are themselves
                # tier-4) can never open as an ordinary gateway focus — a natural-spawn siege
                # cannot reach it any more than the wall behind it (P1 territory: that wall needs
                # a relay, not a door). Skip to the next real missing link.
                from auction.craftax_achievements import tier_of

                if tier_of(door) >= 4:
                    continue
            if rank > 0 and share < DOOR_MIN_SHARE:
                continue
            door_sr = latest_profile.get(door)
            if door_sr is None:
                # unknown SR = closed only at rank 0 (original semantics); deeper ranks need a
                # measured reading — never open a possibly-hallucinated link name as a focus.
                if rank == 0:
                    return door
                continue
            if float(door_sr) < self.th.door_min_sr:
                return door
        return None

    def _yield_to_momentum(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """v6fix10 ②: a focus whose last TWO readings gained >= yield_enter_pp combined while its
        SR is inside the learnability band has been taken over by natural momentum — park it in the
        WATCH registry (all siege privileges withdraw by construction: it leaves ``foci``, which is
        what seats / force-activation / quota / gap feeding key off). resume_lock readings of
        hysteresis after a resume prevent the fix9-#2 oscillation family."""
        kept = []
        for foc in self._nb.get("foci", []):
            # v7: a live relay never yields — its held-out sits at 0 through every mid rung, so a
            # yield could only misfire on noise; the rung machine owns its lifecycle until sewn.
            if self._relay_active(foc):
                kept.append(foc)
                continue
            lock = int(foc.get("resume_lock", 0) or 0)
            if lock > 0:
                foc["resume_lock"] = lock - 1
                kept.append(foc)
                continue
            hist = [h for h in foc.get("sr_history", []) if isinstance(h, (int, float))]
            sr = hist[-1] if hist else None
            # momentum = two consecutive STRICTLY-RISING readings summing >= yield_enter_pp; an
            # oscillation (75 -> 40) or a flat-then-jump is not a climb the curriculum owns yet.
            d1 = (hist[-1] - hist[-2]) if len(hist) >= 2 else None
            d2 = (hist[-2] - hist[-3]) if len(hist) >= 3 else None
            gain2 = (d1 + d2) if (d1 is not None and d2 is not None) else None
            rising = d1 is not None and d2 is not None and d1 > 0 and d2 > 0
            in_band = sr is not None and self.th.learnable_lo <= float(sr) < self.th.graduate_sr
            if in_band and rising and gain2 is not None and gain2 >= self.th.yield_enter_pp:
                sl = str(foc.get("skill", "")).lower()
                w = dict(foc)
                w["watch_since"] = session_idx
                # ②' interface fixes: gap state suspends cleanly (fresh count on resume), and the
                # enabler budget freezes automatically (watch entries never pass _update_focus_stall).
                w["gap_sessions"] = 0
                w["gap_forced"] = False
                w["gap_stall"] = 0
                w.pop("gap_force_baseline", None)
                w["frozen_sessions"] = 0
                self._nb.setdefault("watch", {})[sl] = w
                self._nb.setdefault("history", []).append(
                    {"session": session_idx, "event": "focus_yielded_watch", "focus": sl,
                     "sr": sr, "gain_2readings": gain2}
                )
                self.last_yield = (
                    f"{sl} YIELDED to natural momentum (last 2 readings +{gain2:.0f}pp, SR "
                    f"{sr:.0f}% inside the learnable band) -> WATCH: siege privileges withdrawn, "
                    f"the normal curriculum owns it; siege resumes only if it stalls below "
                    f"{self.th.graduate_sr:.0f}%"
                )
            else:
                kept.append(foc)
        self._nb["foci"] = kept

    def _process_watch(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """v6fix10 ②: the WATCH registry's per-session pass — record the reading, GRADUATE a
        watcher that holds >= graduate_sr (same exit as an active focus), and RESUME a stalled one
        (two consecutive gains < yield_stall_pp while still below graduate_sr) back into ``foci``
        when a slot is free (a resume needs a slot but not the expand gate: it is reclaiming)."""
        watch = self._nb.get("watch") or {}
        for sl, w in list(watch.items()):
            if w.get("watch_since") == session_idx:
                continue  # yielded THIS session: its current reading is already in sr_history
            sr = latest_profile.get(sl)
            hist = [h for h in w.get("sr_history", []) if isinstance(h, (int, float))]
            if sr is not None:
                hist.append(float(sr))
                w["sr_history"] = hist[-max(2 * self.th.slope_window, 12):]
                if float(sr) >= self.th.graduate_sr:
                    w["consecutive_graduate"] = int(w.get("consecutive_graduate", 0)) + 1
                else:
                    w["consecutive_graduate"] = 0
                if int(w.get("consecutive_graduate", 0)) >= self.th.graduate_consecutive:
                    self._graduate_entry(w, session_idx, latest_profile)
                    del watch[sl]
                    continue
            stalled = (
                len(hist) >= 3
                and (hist[-1] - hist[-2]) < self.th.yield_stall_pp
                and (hist[-2] - hist[-3]) < self.th.yield_stall_pp
                and hist[-1] < self.th.graduate_sr
            )
            # v7fix5.0 P2: an access-capped watcher LOOKS permanently stalled (its held-out is
            # pinned by the frontier, not by siege attention) — resuming it would re-take the
            # seat, re-park next reading, and oscillate forever. Hold it in WATCH while the
            # certified cap lasts; when the frontier link rises past ACCESS_CAP_REACH the cap
            # disappears from the next note_access_caps() feed and resume re-engages by itself.
            _cap50 = self._access_cap(sl)
            if stalled and _cap50 and _cap50.get("certified"):
                continue
            # v7fix5.2 P0: a frontier-STARVED parked wall (park_event set by _retire_or_park)
            # resumes only when the frontier it was parked ON has actually MOVED — the wall's own
            # stall readout is structural (held-out pinned by the frontier), not a resume signal.
            # "Moved" = current frontier SR >= park snapshot + focus_improve_pp (the same +3pp
            # evidence bar the blacklist escape hatch uses). If the whole cap dissolved
            # (reach >= 35% -> no cap in this session's feed), the hold stands down and the
            # ordinary stall-resume below applies.
            if stalled and _cap50 and w.get("park_event"):
                _base = w.get("park_frontier_sr")
                _cur = latest_profile.get(str(w.get("park_frontier") or "").lower())
                _moved = (
                    _base is not None and _cur is not None
                    and float(_cur) >= float(_base) + self.th.focus_improve_pp
                )
                if not _moved:
                    continue
            if stalled and len(self._nb.get("foci", [])) < self.th.max_focus:
                f = dict(w)
                f.pop("watch_since", None)
                # v7fix5.2: park bookkeeping does not ride back into an active focus (a later
                # re-park re-snapshots against the then-current frontier).
                f.pop("park_event", None)
                f.pop("park_frontier", None)
                f.pop("park_frontier_sr", None)
                f["resume_lock"] = self.th.resume_lock_readings
                f["gap_sessions"] = 0
                f["gap_forced"] = False
                f["gap_stall"] = 0
                f.pop("gap_force_baseline", None)
                f["frozen_sessions"] = 0
                self._nb.setdefault("foci", []).append(f)
                del watch[sl]
                self._nb.setdefault("history", []).append(
                    {"session": session_idx, "event": "focus_resumed_from_watch", "focus": sl,
                     "sr": hist[-1] if hist else None}
                )
                self.last_resume = (
                    f"{sl} RESUMED from WATCH (natural momentum stalled: last readings "
                    f"{[round(h) for h in hist[-3:]]}, still < {self.th.graduate_sr:.0f}%) -> "
                    f"the siege retakes it (resume_lock {self.th.resume_lock_readings} readings)"
                )
            # else: keep watching (a stalled watcher with no free slot retries next session).

    def _update_highwater(self, latest_profile: dict[str, float], session_idx: int | None = None) -> None:
        """v6fix10 ⑤: ratchet every skill's confirmed peak SR at/above highwater_sr into the registry.

        v6fix10.1 hazard-4: confirmation takes TWO consecutive readings >= highwater_sr and ratchets
        their MIN (the level the skill actually HELD), mirroring the graduation gate's x2 rule. A
        single burst overshoot (a lockhole snapshot at 61 on a skill that settles at 45) used to
        poison the peak permanently: 45 <= 61-15 reads as FORGETTING forever and rehearsal pumps a
        level the student never sustained. A sub-bar reading clears the pending candidate."""
        hw = self._nb.setdefault("highwater", {})
        pend = self._nb.setdefault("highwater_pending", {})
        for skill, sr in latest_profile.items():
            sl = str(skill).lower()
            if sr is None or float(sr) < self.th.highwater_sr:
                pend.pop(sl, None)
                continue
            prev = pend.get(sl)
            if (
                isinstance(prev, (list, tuple)) and len(prev) == 2
                and (session_idx is None or int(prev[0]) < int(session_idx))
            ):
                confirmed = min(float(prev[1]), float(sr))
                if confirmed > float(hw.get(sl, 0.0)):
                    hw[sl] = round(confirmed, 2)
            pend[sl] = [int(session_idx) if session_idx is not None else -1, float(sr)]

    def highwater_forgetting(self, latest_profile: dict[str, float]) -> set[str]:
        """v6fix10 ⑤: high-water skills currently sitting >= highwater_drop_pp below their peak —
        the FORGETTING source that catches erosion the profile-log's fixed peak/drop bars miss
        (fix8: iron_sword 67.8 -> 59.2, an 8.6pp slide under the 20pp bar, guarded by nobody)."""
        latest_profile = {str(k).lower(): v for k, v in (latest_profile or {}).items()}
        out: set[str] = set()
        for sl, peak in (self._nb.get("highwater") or {}).items():
            sr = latest_profile.get(sl)
            if sr is not None and float(sr) <= float(peak) - self.th.highwater_drop_pp:
                out.add(sl)
        return out

    def _graduate_entry(
        self, foc: dict, session_idx: int, latest_profile: dict[str, float]
    ) -> None:
        """The graduation body shared by active foci (fix8 ①) and WATCH entries (fix10 ②):
        record the experience (status stays 'progress'), move the wall into maintenance, and
        protect wall+links. The caller removes ``foc`` from its own container."""
        from auction.craftax_achievements import family_of

        skill = str(foc.get("skill") or "").lower()
        sr = latest_profile.get(skill)
        links = [
            l.get("skill") for l in foc.get("prereq_tree", []) if isinstance(l.get("skill"), str)
        ]
        category = "combat_milestone" if family_of(skill) == "COMBAT" else "enabler"
        self._upsert_experience(
            session_idx, skill, links, sr if sr is not None else float(self.th.graduate_sr),
            category, style_note=str(foc.get("style_note", "")), status="progress",
        )
        self._nb.setdefault("maintenance", {})[skill] = {
            "graduated_session": session_idx,
            "sr_at_graduation": sr,
            "links": [str(l).lower() for l in links if l],
            "style_note": str(foc.get("style_note", "")),
        }
        protect = {skill} | {str(l).lower() for l in links if l}
        self._nb["protected_set"] = sorted(set(self._nb.get("protected_set", [])) | protect)
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "focus_graduated_maintenance", "focus": skill,
             "sr": sr, "held_snapshots": int(foc.get("consecutive_graduate", 0))}
        )
        self.last_graduation = (
            f"{skill} GRADUATED to maintenance (SR {sr}%, held >= {self.th.graduate_sr}% for "
            f"{foc.get('consecutive_graduate')} snapshots — fast climber, not a real wall; "
            "siege privileges withdrawn, rehearsal holds it)"
        )

    def _merge_style_notes(self, proposed_foci: list[dict], session_idx: int | None = None) -> None:
        """§3.1: copy each proposed focus's fresh ``style_note`` onto the matching ACTIVE focus dict.

        Only non-empty notes overwrite (a session where the LLM says nothing new keeps the prior note,
        so know-how accumulates rather than being blanked). Matching is by skill name, so this naturally
        follows the same dedup-by-target rule as the experience log.

        v6fix7 P1c (AutoManual-lite lifecycle): each proposed focus also carries ``evidence_check``
        (supported / contradicted / no_evidence — the modeler's self-audit of its tactic against this
        session's REAL evidence). The notebook turns that into a code-tracked note status:
          - supported     -> note_status=active, note_last_supported_session=now;
          - contradicted  -> the note MUST be rewritten (materially different — same-tactic rephrase
                             rejected, like ladder L3); until rewritten it renders as CONTRADICTED;
          - no_evidence   -> ages toward STALE after note_stale_sessions unsupported sessions.
        """
        if not proposed_foci:
            proposed_foci = []
        by_skill = {
            str(pf.get("skill", "")).lower(): pf
            for pf in proposed_foci
            if isinstance(pf, dict) and pf.get("skill")
        }
        for foc in self._nb.get("foci", []):
            pf = by_skill.get(str(foc.get("skill", "")).lower())
            note = str(pf.get("style_note", "")).strip() if pf else ""
            evidence = str(pf.get("evidence_check", "")).strip().lower() if pf else ""
            old_note = str(foc.get("style_note", ""))

            # v6fix9 P2: store the (gate-checked) failure attribution alongside the note — the
            # journal renders it, so the causal claim the modeler committed to stays visible and
            # cross-session drift ("pressure" one week, "resources" the next) is auditable.
            if pf is not None and isinstance(pf.get("failure_attribution"), dict):
                foc["failure_attribution"] = dict(pf["failure_attribution"])

            # --- lifecycle bookkeeping (runs even when the note itself is empty) ---
            if session_idx is not None and pf is not None:
                if evidence == "supported":
                    foc["note_last_supported_session"] = session_idx
                    foc["note_status"] = "active"
                elif evidence == "contradicted":
                    foc["note_status"] = "contradicted"
                else:  # no_evidence / missing -> age toward stale
                    anchor = foc.get("note_last_supported_session")
                    if anchor is None:
                        anchor = foc.get("started_session") or session_idx
                    if old_note and (session_idx - int(anchor)) >= self.th.note_stale_sessions \
                            and foc.get("note_status") != "contradicted":
                        foc["note_status"] = "stale"

            if not note:
                continue
            # v6fix7 P1a L3 + P1c contradicted: when a REVISION is demanded (ladder >= 3, or the
            # modeler itself judged the tactic contradicted by evidence), a note materially the same
            # as the current one is rejected (kept old + history event) — "refine" is not enough.
            revision_required = int(foc.get("ladder_level", 0)) >= 3 or \
                foc.get("note_status") == "contradicted"
            if revision_required and self._notes_similar(note, old_note):
                self._nb.setdefault("history", []).append(
                    {"session": session_idx if session_idx is not None else self._nb.get("last_session"),
                     "event": "tactic_revision_rejected", "focus": foc.get("skill")}
                )
                continue
            if foc.get("note_status") == "contradicted" and note != old_note:
                # rewrite accepted: back to active (still unsupported until evidence says otherwise).
                foc["note_status"] = "active"
            foc["style_note"] = note

    def _record_experience(self, session_idx: int, latest_profile: dict[str, float]) -> None:
        """§2① + §2.5: incrementally record/UPDATE each active focus's success experience.

        For every active focus whose SR has risen by >= ``record_delta_pp`` versus the LAST time it was
        recorded (or first crossing above 0-ish), write/update (DEDUP BY TARGET) a verified_chains entry
        categorised combat_milestone vs enabler. Noise jitter below the delta records nothing -> no
        near-duplicate spam. The focus is NOT retired here (that is stall-driven, §2⑤); a focus keeps
        being attacked and its evidence keeps upgrading ("15% -> 40%, getting stronger").

        NOTE (ordering): this runs BEFORE new foci are opened in apply_llm_update, so a focus is first
        recorded the session AFTER it opens (once it has a real SR reading as an active focus) — this
        deliberately avoids recording a focus the same session it is proposed but before any training.
        """
        from auction.craftax_achievements import family_of

        for foc in self._nb.get("foci", []):
            skill = foc.get("skill")
            if not isinstance(skill, str):
                continue
            sr = latest_profile.get(skill.lower())
            if sr is None:
                continue
            last = foc.get("last_recorded_sr")
            if last is not None and sr < last + self.th.record_delta_pp:
                continue  # not enough new progress — skip (dedup/anti-noise)
            links = [
                link.get("skill")
                for link in foc.get("prereq_tree", [])
                if isinstance(link.get("skill"), str)
            ]
            category = "combat_milestone" if family_of(skill.lower()) == "COMBAT" else "enabler"
            # v6fix7 P1a (#8 fix): a +delta record is PROGRESS, not conquest. It no longer touches
            # the protected_set and its chain entry carries status='progress' — only
            # _verify_conquests (mastered_sr held for conquest_consecutive snapshots) upgrades an
            # entry to 'verified', protects it, and reports a conquest.
            self._upsert_experience(
                session_idx, skill.lower(), links, sr, category,
                style_note=str(foc.get("style_note", "")), status="progress",
            )
            foc["last_recorded_sr"] = sr

    def _upsert_experience(
        self, session_idx: int, target: str, links: list[str], sr: float, category: str,
        style_note: str = "", status: str = "progress",
    ) -> None:
        """Insert or, if ``target`` already recorded, UPDATE its verified_chains entry (dedup by target).

        ``style_note`` (§3.1) is the transferable attack know-how. On UPDATE it is overwritten only when
        the new note is non-empty, so a silent session keeps the prior know-how instead of erasing it.
        v6fix7 (#8): ``status`` distinguishes 'progress' (delta-recorded, NOT reusable as a tier4 base)
        from 'verified' (conquest gate passed). A 'verified' entry is never downgraded."""
        chains = self._nb.setdefault("verified_chains", [])
        evidence = f"held-out SR {sr}% (recorded at s{session_idx})"
        note = str(style_note or "").strip()
        for c in chains:
            if isinstance(c, dict) and str(c.get("target", "")).lower() == target:
                # update in place: keep first_recorded_sr, refresh the rest, merge links.
                c["last_recorded_sr"] = sr
                c["last_recorded_session"] = session_idx
                c["category"] = category
                merged_links = list(dict.fromkeys([*c.get("links", []), *links]))
                c["links"] = merged_links
                c["evidence"] = evidence
                if status == "verified" or c.get("status") == "verified":
                    c["status"] = "verified"  # never downgrade a verified conquest
                else:
                    c["status"] = "progress"
                if note:  # keep prior know-how if this session added none
                    c["style_note"] = note
                return
        chains.append({
            "target": target,
            "links": links,
            "category": category,
            "first_recorded_sr": sr,
            "last_recorded_sr": sr,
            "last_recorded_session": session_idx,
            "evidence": evidence,
            "style_note": note,
            "status": status,
        })

    # ---- the one public write path: apply an LLM-proposed update through B-layer rules ----------

    def apply_llm_update(
        self,
        session_idx: int,
        latest_profile: dict[str, float],
        proposed: dict | None,
        num_snapshots: int = 999,
        forensics: dict | None = None,
        chain_incomplete: set[str] | None = None,
    ) -> dict:
        """Fold one LLM proposed-update into the notebook, enforcing every B-layer rule, then persist.

        ``proposed`` (the A-layer output; may be None/empty if the LLM produced nothing usable) may
        contain EITHER the multi-focus form or the legacy single-focus form:
          - multi (§2.6):  "foci": [ {"skill", "prereq_tree":[{"skill","role"},...]}, ... ]
          - legacy:        "focus": <skill>, "prereq_tree": [ {"skill","role"}, ... ]
        The legacy form is normalised into a one-element ``foci`` list here, so old callers/tests keep
        working. The LLM supplies skill + role only; the CODE fills every mastery flag/state.

        B-layer enforcement order (§2.6 multi-focus):
          1. refresh existing link flags from live SR (LLM never sets mastery),
          2. §2① incrementally record each focus's success experience (dedup by target, categorised),
          3. per-focus stall bookkeeping, then retire any STALLED focus (never by SR threshold),
          4. reconcile the LLM's desired foci set: keep still-wanted existing foci; OPEN a new focus
             only if MATURE AND it passes §3.2 scope AND the §2.6 expand gate (free slot + one focus
             already going well) allows it,
          5. attach each accepted focus's prereq_tree, recomputing every link's mastery flag from SR.

        ``num_snapshots`` = how many held-out snapshots exist so far; used by the maturity gate.

        Returns the resulting notebook snapshot.
        """
        proposed = self._normalise_proposal(proposed)
        latest_profile = {k.lower(): v for k, v in (latest_profile or {}).items()}
        self._last_profile = latest_profile
        # reset per-call diagnostics (the caller reads these to log).
        self.last_focus_decision = None
        self.last_conquest = None
        self.last_graduation = None
        self.last_budget_retire = None
        self.last_auto_open = None
        self.last_yield = None
        self.last_resume = None
        self.last_door_sub = None
        self.last_attrib_arm = None
        self.last_relay_open = None
        self.last_chain_autofill = None  # v7fix4 P1: entrance autofill note (caller logs it)
        self.last_park = None         # v7fix5.2 P0: retire->park routing note (caller logs it)
        self.last_access_auto = None  # v7fix5.2 P1: access-root nomination note (caller logs it)
        self.last_probe_decision = None  # v7fix5.5: probe accept/reject receipt (caller logs it)
        # v6fix10 per-call evidence context (① door gate / ⑦ chain-incomplete gate). Held only for
        # the duration of this call; both come from the student's OWN failure forensics, no priors.
        # v6fix10.1 hazard-3a: ``forensics is None`` means the caller has no chain log at all (old
        # tests / degraded wiring) — the admission-deferral gate then stands down, because deferring
        # would wait forever for forensics that can never arrive. gen_manager passes a dict (possibly
        # empty) whenever the chain log exists, so production always runs the gate.
        self._call_forensics = {str(k).lower(): v for k, v in (forensics or {}).items()}
        self._call_forensics_provided = forensics is not None
        self._call_chain_incomplete = {str(s).lower() for s in (chain_incomplete or set())}

        # 1. existing links: flags come from data, not the model.
        self._refresh_link_flags(latest_profile)

        # 1b. §3.1 self-style: fold this session's fresh notes onto the EXISTING active foci first, so
        #     _record_experience carries the LATEST know-how into verified_chains (not last session's).
        #     A newly-opened focus is merged again in step 6 (it isn't active yet here).
        self._merge_style_notes(proposed.get("foci") or [], session_idx=session_idx)

        # 2. §2① incremental success-experience recording (replaces the old conquered-retire gate).
        self._record_experience(session_idx, latest_profile)

        # 3. per-focus stall (frozen counter + ladder level), conquest verification (#8), then the
        #    v6fix8 exits — fast-climb graduation (①) and enabler-budget retirement (⑤) — then retire
        #    truly frozen foci (L4; frees slots). Exit precedence: CONQUERED (>=70x2) outranks
        #    GRADUATED (>=50x2) outranks budget/stall retirement.
        self._update_focus_stall(latest_profile)
        self._verify_conquests(session_idx, latest_profile)
        self._graduate_fast_climbers(session_idx, latest_profile)
        self._retire_budget_exhausted(session_idx)
        self._retire_stalled_foci(session_idx)
        # v6fix10: ⑤ high-water ratchet, then ② yield-to-momentum (runs AFTER graduation so a >=50
        # climber exits as GRADUATED, never as yielded) and the watch registry's graduate/resume
        # pass (before reconcile, so a resumed focus reclaims its slot ahead of new proposals).
        self._update_highwater(latest_profile, session_idx=session_idx)
        self._yield_to_momentum(session_idx, latest_profile)
        self._process_watch(session_idx, latest_profile)

        # 4. reconcile the desired foci set against the B-layer gates.
        # v7fix3 P1/P0: the tier-locked journal mirror covers LAST session only — clear it on
        # every call (reconcile re-fills it; a no-proposal session must not keep stale nagging).
        self._nb["tier_locked_last"] = []
        mature = self._student_is_mature(latest_profile, num_snapshots)
        # 3d. v7fix5.2 P1: deterministic access-root nomination — BEFORE the LLM reconcile so the
        #     root gets first pick of a naturally-free seat (never evicts, never bypasses a gate);
        #     the LLM keeps its freedom on the remaining seats, and the 4b auto-open still fills
        #     whatever is left after both.
        if mature:
            self._access_auto_nominate(session_idx, latest_profile)
        proposed_foci = proposed.get("foci") or []
        if not proposed_foci:
            self.last_focus_decision = "no_proposal"
        elif not mature:
            n_decent = sum(
                1 for v in latest_profile.values()
                if v is not None and v >= self.th.maturity_skill_sr
            )
            self.last_focus_decision = (
                f"immature (snapshots={num_snapshots}<{self.th.maturity_min_snapshots} or "
                f"decent_skills={n_decent}<{self.th.maturity_min_mastered}) -> NORMAL_EARLY, no siege"
            )
        else:
            self._reconcile_foci(session_idx, latest_profile, proposed_foci)

        # 4b. v6fix8 ②: MULTI-FOCUS HARD GATE — fix7 left the second focus to the LLM's mood
        #     ("Prefer a COMBAT wall...") and it never proposed one in 17 sessions. When the expand
        #     gate is satisfied and a slot is free but the LLM's own proposal did not fill it, the
        #     code opens the top viable candidate from the modeler's ranked_walls itself (COMBAT
        #     candidates first while no combat focus is active). The LLM keeps agency over WHAT is
        #     ranked; the code guarantees the slot does not sit empty.
        if mature:
            self._auto_open_from_ranked(session_idx, latest_profile, proposed.get("ranked_walls"))
            # 4c. v7fix4.2 (2)+(3): the deep-wall relay trigger — arm/advance the journal
            #     directive, and force-open the top candidate after K ignored decisions.
            #     After auto-open so a slot it just filled is seen; before prereq attach so a
            #     forced relay gets its entrance chain autofilled the same session.
            self._relay_trigger_tick(session_idx, latest_profile, proposed_foci)

        # 5. attach each accepted focus's prereq_tree (code owns every mastery flag).
        self._attach_prereq_trees(latest_profile, proposed_foci)

        # 6. §3.1 self-style: fold the LLM's fresh attack know-how onto EVERY current focus (existing +
        #    just-opened). Done last so a newly-opened focus also captures its note this session; it is
        #    then carried into verified_chains by _record_experience NEXT session. LLM owns this text
        #    (A-layer); code only stores the latest non-empty note per target.
        self._merge_style_notes(proposed.get("foci") or [], session_idx=session_idx)

        # 6b. v7fix5.5 PROBE-AS-TOOL: fold the modeler's optional measurement ask through the
        #     probe gates (after reconcile so the relay foci set is current). The pending probe
        #     is executed on the MAIN thread between sessions (run_dicode); nothing here touches
        #     training state.
        if mature:
            self._admit_probe_request(proposed.get("probe_request"), session_idx)

        # 7. v6fix10 ④ attribution->form shortcut: a VERIFIED access-blocked attribution
        #    (chain_unreached / resource_shortfall whose key link is still behind the door line)
        #    forces DEPTH immediately via required_form — no waiting for the gap gate's 3 readings.
        #    One diagnostic drill decision is allowed (required_form checks siege_sessions >= 1).
        for foc in self._nb.get("foci", []):
            # v7: a live relay wall's attribution WILL verify as access-blocked (that is why it
            # is a relay) — forcing DEPTH would ban the R0/mid-rung drills the campaign exists
            # to run. The rung machine owns the form until the relay sews; ④ resumes after.
            if self._relay_active(foc):
                foc["attrib_depth_required"] = False
                continue
            a = foc.get("failure_attribution") or {}
            key = a.get("key_missing_link")
            key_sr = latest_profile.get(str(key).lower()) if isinstance(key, str) else None
            foc["attrib_depth_required"] = bool(
                a.get("verified")
                and str(a.get("class", "")).lower() in ("chain_unreached", "resource_shortfall")
                and key is not None
                and (key_sr is None or float(key_sr) < self.th.door_min_sr)
            )
            # v6fix10.1 hazard-1: the moment the shortcut becomes OPERATIVE (past the one allowed
            # diagnostic drill), latch gap_forced so the P3 early-stop counts for this wall too.
            # ④ stops drills from decision 2, so the gap gate's trained>=90 readings can never
            # accumulate and gap_forced would never latch on its own — leaving a ④-path COMBAT
            # wall (no ⑤ budget) with NO early-stop exit at all. fix9's kobold survived only
            # because STYLE_REJECTED fired at s29; that fuse must exist on this path as well.
            if (
                foc["attrib_depth_required"]
                and int(foc.get("siege_sessions", 0)) >= 1
                and not foc.get("gap_forced")
            ):
                foc["gap_forced"] = True
                held = latest_profile.get(str(foc.get("skill", "")).lower())
                foc["gap_force_baseline"] = float(held) if held is not None else None
                foc["gap_stall"] = 0
                self._nb.setdefault("history", []).append(
                    {"session": session_idx, "event": "attrib_shortcut_armed_earlystop",
                     "focus": foc.get("skill"), "key": key, "held": held}
                )
                self.last_attrib_arm = (
                    f"{foc.get('skill')}: verified access-blocked attribution (key={key}) went "
                    f"operative -> gap_forced LATCHED (P3 early-stop now counts; baseline "
                    f"held-out={held})"
                )

        # v6fix10.1 hazard-3a: prune pending-track entries nobody re-proposed/ranked for
        # PENDING_TRACK_TTL sessions. Walls that are gateway_for of a live door siege are exempt —
        # they are tracked through the gateway on purpose (fresh forensics for their return).
        pend = self._nb.get("pending_track") or {}
        if pend:
            gated = {
                str(f.get("gateway_for")).lower()
                for f in list(self._nb.get("foci", [])) + list((self._nb.get("watch") or {}).values())
                if isinstance(f, dict) and f.get("gateway_for")
            }
            for sl in list(pend):
                entry = pend[sl] if isinstance(pend[sl], dict) else {}
                if sl in gated:
                    continue
                if session_idx - int(entry.get("session", session_idx)) > PENDING_TRACK_TTL:
                    del pend[sl]

        self._nb["last_session"] = session_idx
        self._save()
        return self.snapshot()

    @staticmethod
    def _normalise_proposal(proposed: dict | None) -> dict:
        """Normalise a proposal into the multi-focus ``{"foci": [{skill, prereq_tree}, ...]}`` form.

        Accepts the legacy single-focus ``{"focus", "prereq_tree"}`` shape and wraps it into a
        one-element foci list, so old callers/tests need no change.
        """
        proposed = proposed or {}
        ranked = proposed.get("ranked_walls") or []
        probe = proposed.get("probe_request")  # v7fix5.5: ride through every shape
        if proposed.get("foci"):
            return proposed
        focus = proposed.get("focus")
        if isinstance(focus, str) and focus.strip():
            return {"foci": [{
                "skill": focus,
                "prereq_tree": proposed.get("prereq_tree") or [],
                "style_note": proposed.get("style_note", ""),  # carry legacy top-level note through
            }], "ranked_walls": ranked, "probe_request": probe}
        return {"foci": [], "ranked_walls": ranked, "probe_request": probe}

    def _reconcile_foci(
        self, session_idx: int, latest_profile: dict[str, float], proposed_foci: list[dict]
    ) -> None:
        """§2.6: keep still-wanted existing foci; open NEW foci through maturity/scope/expand/cap gates."""
        active = {f["skill"].lower() for f in self._nb.get("foci", []) if isinstance(f.get("skill"), str)}
        watching = {str(s).lower() for s in (self._nb.get("watch") or {})}
        decisions: list[str] = []
        tier_locked_now: list[str] = []  # v7fix3 P1: mirrored into the journal (LLM-visible)
        for pf in proposed_foci:
            skill = pf.get("skill")
            if not isinstance(skill, str) or not skill.strip():
                continue
            sl = skill.lower()
            if sl in active:
                # v7fix3 P2: an ACTIVE ordinary focus re-proposed WITH relay_r0_floor is an
                # UPGRADE ask, not a duplicate — v7fix2's pigman deadlock was exactly this line
                # swallowing the relay answer as `kept()` (same disease as v7 first-run's ⑦-before-
                # relay-branch, one gate over). The attach itself is ruled by _attach_relay.
                _r0_up = pf.get("relay_r0_floor")
                if self.th.relay_attach and isinstance(_r0_up, (int, float)) and int(_r0_up) >= 1:
                    decisions.append(
                        self._attach_relay(sl, int(_r0_up), session_idx, latest_profile)
                    )
                else:
                    decisions.append(f"kept({sl})")
                continue
            # v6fix10 ②: a WATCH wall is already owned by natural momentum — do not reopen it;
            # _process_watch resumes it automatically the moment its momentum stalls.
            if sl in watching:
                decisions.append(
                    f"watching({sl}: yielded to natural momentum — resumes on stall, not on request)"
                )
                continue
            # v7fix3 P1: a VERY_ADVANCED (tier-4) wall lives on deep floors — a natural-spawn siege
            # can never reach it (v7fix2: pigman opened ordinary at s21, held-out 0.0 x6 readings,
            # and its 0% blocked every other campaign via the expand gate). Such a wall may open
            # ONLY as a spawn-anneal relay; the refusal teaches the exact re-proposal format.
            _r0_pf = pf.get("relay_r0_floor")
            _relay_asked = isinstance(_r0_pf, (int, float)) and int(_r0_pf) >= 1
            if self.th.tier4_relay_only and not _relay_asked:
                from auction.craftax_achievements import tier_of

                if tier_of(sl) >= 4:
                    decisions.append(
                        f"tier_locked({sl}: a VERY_ADVANCED wall lives on deep floors — a "
                        f"natural-spawn siege cannot reach it and would burn the slot at 0%; "
                        f"re-propose THIS wall WITH relay_r0_floor=<its floor> to open a "
                        f"spawn-anneal relay campaign instead)"
                    )
                    tier_locked_now.append(sl)
                    continue
            # v7fix4 P1: a wall whose native habitat is floor DEEP_WALL_RELAY_FLOOR+ may open ONLY
            # as a relay. v7fix3 post-mortem: defeat_lizard (floor 3, Sewers) opened ORDINARY at
            # s19, its FM drills placed the lizard shallow, and the fake wins disarmed ⑦ for the
            # rest of the run. Floor-3+ doors (enter_sewers on) have NEVER opened naturally in any
            # 150-session arm incl. base — an ordinary siege of such a wall is unwinnable in
            # held-out by construction. Floor-2 walls (gnome class) are deliberately NOT locked:
            # fix8 proved they crack via the fix10 door-gate + natural exposure (26.4% final).
            if self.th.wall_floor_anchor and not _relay_asked:
                from auction.craftax_achievements import native_floor_of

                _nf = native_floor_of(sl)
                if _nf >= self.th.deep_wall_relay_floor:
                    # v7fix4.2 (1) AUTOCONVERT: the refusal used to be the end of the road — and
                    # since no LLM ever cold-starts a relay_r0_floor proposal, it was the end of
                    # ALL tier-3 attacks (fix4-run s77: zero deep proposals, zero relays). Now the
                    # ordinary ask becomes the relay ask IN PLACE and falls through to the explicit
                    # -relay pipeline below (⑦ exemption, expand exemption, relay_max capacity with
                    # its deep-wall fall-through closure, r0 anchoring): every existing safety net
                    # applies unchanged. Convert only when the ladder is actually startable and
                    # useful: tier-4 stays tier_locked upstream (never reaches here while
                    # tier4_relay_only is on; the explicit tier check below guards the off case),
                    # and a wall with real held-out wins keeps the plain lock (it needs no spawn
                    # ladder — the attach path applies the same evidence rule).
                    _sr_now = latest_profile.get(sl)
                    _convertible = self.th.deep_wall_autoconvert
                    if _convertible:
                        from auction.craftax_achievements import tier_of

                        _convertible = tier_of(sl) < 4 and (
                            _sr_now is None or float(_sr_now) <= self.th.zero_win_max_sr
                        )
                    _n_relays_now = sum(
                        1 for f in self._nb.get("foci", []) if self._relay_active(f)
                    )
                    if _convertible and _n_relays_now < self.th.relay_max \
                            and self._may_open_new_focus(latest_profile, relay=True):
                        pf["relay_r0_floor"] = _nf
                        decisions.append(
                            f"relay_converted({sl}: it inhabits dungeon floor {_nf} — deep walls "
                            f"open ONLY as spawn-anneal relay campaigns, so your ordinary "
                            f"proposal was auto-converted into one (R0 anchored to floor {_nf}); "
                            f"keep it in your foci and supply its prereq_tree/style_note as for "
                            f"any relay wall)"
                        )
                        self._nb.setdefault("history", []).append(
                            {"session": session_idx, "event": "relay_converted",
                             "focus": sl, "r0": _nf}
                        )
                        # NO continue: falls through to the explicit-relay open path below.
                    else:
                        _busy = (
                            " (the relay slot/foci are busy right now — a slot frees when a "
                            "campaign sews or retires; keep proposing it)"
                            if _convertible else ""
                        )
                        decisions.append(
                            f"deep_locked({sl}: it inhabits dungeon floor {_nf} and no "
                            f"natural-spawn curriculum has ever reached floor "
                            f"{self.th.deep_wall_relay_floor}+ — an ordinary siege cannot win it "
                            f"in held-out; re-propose THIS wall WITH relay_r0_floor=<any floor "
                            f">= 1> to open a spawn-anneal relay campaign (the system anchors R0 "
                            f"to its habitat floor {_nf}){_busy}"
                        )
                        tier_locked_now.append(sl)
                        continue
            # v6fix10 ⑦: the reported chain is complete-in-failures yet the wall never wins — a
            # prerequisite is MISSING from the chain; re-attacking on the same chain is refused.
            # v7fix1 (first-run s23 post-mortem): a RELAY proposal is EXEMPT — the ⑦ latch used to
            # sit before the relay branch and silently swallowed every deep-wall relay proposal
            # (kobold s11-13, troll s15-23: the LLM's relay answer to ⑦ could never arrive). The
            # rung ladder + R1/R3/R6 replace chain completeness structurally (same family as the
            # door-gate demotion for relays); ⑦ still rules NON-relay re-attacks, and its message
            # now teaches the relay exit for deep-floor prerequisites.
            _r0_req = pf.get("relay_r0_floor")
            relay_requested = isinstance(_r0_req, (int, float)) and int(_r0_req) >= 1
            if sl in self._call_chain_incomplete and not relay_requested:
                decisions.append(
                    f"chain_incomplete({sl}: failures reach your full reported chain yet zero wins "
                    "— the chain is missing an unnamed prerequisite; EXPAND the prereq_tree with "
                    "what else must be true, using EXACT achievement names from the SR profile "
                    "(a step with no achievement name goes in style_note, not the chain); if the "
                    "missing prerequisite is a DEEPER FLOOR, re-propose THIS wall with "
                    "relay_r0_floor=<floor> instead of inventing links)"
                )
                continue
            # a NEW focus: must be a real wall (scope) and there must be room + slack (expand gate).
            if not self._is_valid_focus(sl, latest_profile):
                decisions.append(
                    f"scope_rejected({sl}:SR{latest_profile.get(sl)}>=sat{self.th.saturated_sr})"
                )
                continue
            # v6fix7 P1a: a VERIFIED (conquered) wall still held at mastered SR may not reopen — it
            # is protected ground (rehearsal guards it). Siege it again only if it genuinely slipped.
            if self._is_conquered_and_held(sl, latest_profile):
                decisions.append(f"conquered_held({sl}: verified and SR still >= mastered)")
                continue
            # v6fix8 ①: a GRADUATED (maintenance) wall may be re-sieged only after a REAL collapse —
            # below graduate_sr - resiege_drop. While it holds, rehearsal is the guard, not the siege.
            maint_reason = self._maintenance_block_reason(sl, latest_profile)
            if maint_reason:
                decisions.append(maint_reason)
                continue
            # v6fix7 P1a: retirement aftermath gates — cooldown, blacklist, and "what's different".
            reg = (self._nb.get("retired") or {}).get(sl)
            if reg:
                last_ret = int(reg.get("last_session", -10**9))
                cooldown_left = self.th.cooldown_sessions - (session_idx - last_ret)
                if cooldown_left > 0:
                    # v7fix3.1 audit fix: a RELAY re-proposal after an ORDINARY-siege retirement
                    # is exempt from the cooldown — the ④ gap gate can retire a zero-win wall in
                    # the very session the LLM answers the ★ZERO-WIN hint, and the relay ladder is
                    # a categorically different attack form (the sanctioned exit for depth-blocked
                    # walls), so making it wait 6 sessions re-creates the pigman lockout one gate
                    # over. A campaign that ITSELF stalled out (focus_retired_relay_stalled) gets
                    # no exemption: re-opening the same failed ladder must wait the cooldown.
                    # v7fix4: nor does a sandbox_mismatch retirement — the ladder ran and its
                    # product failed the reality check; the same ladder must not re-run on cooldown
                    # waiver, the chain needs expansion first.
                    if relay_requested and reg.get("last_event") not in (
                        "focus_retired_relay_stalled", "focus_retired_sandbox_mismatch",
                    ):
                        decisions.append(
                            f"cooldown_waived({sl}: relay re-proposal after an ordinary-siege "
                            f"retirement — the spawn ladder is a different attack form; "
                            f"blacklist/new-tactic checks still apply)"
                        )
                    else:
                        decisions.append(
                            f"cooldown_rejected({sl}: {cooldown_left} session(s) left)"
                        )
                        continue
                if self._blacklist_count(reg) >= self.th.blacklist_retirements and not \
                        self._has_new_evidence(sl, reg, latest_profile):
                    decisions.append(
                        f"blacklisted({sl}: retired {reg.get('count')}x, no new chain evidence)"
                    )
                    continue
                new_note = str(pf.get("style_note", "")).strip()
                if not new_note or any(
                    self._notes_similar(new_note, old) for old in reg.get("failed_notes", [])
                ):
                    decisions.append(
                        f"reopen_needs_new_tactic({sl}: proposal's tactic is empty or repeats an "
                        "archived failed tactic — state what is DIFFERENT this time)"
                    )
                    self._nb.setdefault("history", []).append(
                        {"session": session_idx, "event": "reopen_rejected_same_tactic", "focus": sl}
                    )
                    continue
            # v7fix3 P3: a RELAY open is exempt from the "any existing focus >= expand_sr"
            # condition — v7fix2's 0% pigman focus held that condition hostage and locked every
            # new campaign (incl. relays) out until its stall retirement. A relay's fire is
            # already structurally throttled (relay_max=1 + the ③ rung-momentum discount), so it
            # only needs a physically free slot; ordinary opens keep the full expand gate.
            # v7fix5.0 P2b: opening a named ACCESS FRONTIER is likewise (b)-exempt — the wall
            # pinning every focus at 0-20% is exactly the wall whose prerequisite this is.
            _is_frontier = sl in self.access_frontiers()
            if not self._may_open_new_focus(
                latest_profile, relay=relay_requested, access_frontier=_is_frontier,
            ):
                if relay_requested:
                    decisions.append(
                        f"expand_refused({sl}: relay asked but no free focus slot "
                        f"({len(self._nb.get('foci', []))}/{self.th.max_focus}))"
                    )
                else:
                    decisions.append(
                        f"expand_refused({sl}: {len(self._nb.get('foci', []))}/{self.th.max_focus} foci, "
                        f"none>={self.th.focus_expand_sr}%)"
                    )
                continue
            if _is_frontier and not relay_requested:
                decisions.append(
                    f"expand_exempt_access_frontier({sl}: named as a tracked wall's access "
                    f"frontier — expand gate (b) waived, capacity still applied)"
                )
            # v7 SPAWN-ANNEAL RELAY: an explicit LLM proposal carrying relay_r0_floor opens this
            # wall as a backward campaign (R0 spawns AT floor r0, annealing up to natural spawn).
            # The admission deferral and the door gate below are deliberately BYPASSED here: both
            # exist to stop attacks THROUGH a closed door, and a relay does not go through the
            # door — it spawns behind it; reachability is the very thing the rung ladder repairs
            # (v7_design.md §2.2: the door gate demotes to R0-depth evidence for relay walls).
            # Every gate ABOVE (scope / conquered / maintenance / cooldown / blacklist /
            # ⑦ chain-incomplete / expand) applied as usual. Capacity: relay_max campaigns; over
            # capacity the proposal falls through to the normal (non-relay) open path.
            r0 = pf.get("relay_r0_floor")
            if relay_requested:
                n_relays = sum(1 for f in self._nb.get("foci", []) if self._relay_active(f))
                if n_relays >= self.th.relay_max:
                    # v7fix1: a ⑦-latched wall may NOT fall through to a normal open when the
                    # relay slot is taken — ⑦ was skipped above only because a relay was asked for.
                    if sl in self._call_chain_incomplete:
                        decisions.append(
                            f"relay_refused({sl}: {n_relays}/{self.th.relay_max} relay campaign(s) "
                            f"already running, and its chain is ⑦-incomplete so a normal open is "
                            f"also refused — keep the relay proposal, a slot frees when a campaign "
                            f"sews or retires)"
                        )
                        continue
                    # v7fix3.1 audit fix: a tier-4 wall may not fall through to a NORMAL open
                    # either — P1 (tier4_relay_only) checked "not relay-asked" above, so without
                    # this re-check a relay ask made while the relay slot is taken would open the
                    # wall as an ordinary natural-spawn siege: exactly the 0%-focus pigman shape
                    # P1 exists to prevent, sneaking in through the capacity fall-through.
                    if self.th.tier4_relay_only:
                        from auction.craftax_achievements import tier_of

                        if tier_of(sl) >= 4:
                            decisions.append(
                                f"relay_refused({sl}: {n_relays}/{self.th.relay_max} relay "
                                f"campaign(s) already running, and a VERY_ADVANCED wall cannot "
                                f"open as a natural-spawn siege — keep the relay proposal, a "
                                f"slot frees when a campaign sews or retires)"
                            )
                            continue
                    # v7fix4 P1: same fall-through closure for habitat-deep walls — without this
                    # re-check, a relay ask made while the slot is taken would open a floor-3+
                    # wall as an ordinary siege (the exact lizard-s19 shape the deep lock exists
                    # to prevent, sneaking in through the capacity fall-through).
                    if self.th.wall_floor_anchor:
                        from auction.craftax_achievements import native_floor_of

                        if native_floor_of(sl) >= self.th.deep_wall_relay_floor:
                            decisions.append(
                                f"relay_refused({sl}: {n_relays}/{self.th.relay_max} relay "
                                f"campaign(s) already running, and a floor-"
                                f"{native_floor_of(sl)} wall cannot open as a natural-spawn "
                                f"siege — keep the relay proposal, a slot frees when a campaign "
                                f"sews or retires)"
                            )
                            continue
                    decisions.append(
                        f"relay_refused({sl}: {n_relays}/{self.th.relay_max} relay campaign(s) "
                        f"already running — treating this as a normal focus proposal)"
                    )
                    # v7fix3 P3: the expand gate above was passed WITH the relay exemption; a
                    # fall-through to a NORMAL open must re-pass the ordinary gate — otherwise
                    # carrying relay_r0_floor would become a free expand-gate bypass.
                    # (v7fix5.0: the access-frontier exemption is kept — it is candidate-keyed,
                    # not lane-keyed, so the fall-through does not change its justification.)
                    if not self._may_open_new_focus(
                        latest_profile, relay=False, access_frontier=_is_frontier,
                    ):
                        decisions.append(
                            f"expand_refused({sl}: {len(self._nb.get('foci', []))}/"
                            f"{self.th.max_focus} foci, none>={self.th.focus_expand_sr}%)"
                        )
                        continue
                else:
                    # v7fix4 P1: R0 is a fact about the world, not a proposal — anchor it to the
                    # wall's native habitat floor (the v7fix3 lizard relay annealed from floor 2
                    # while lizards inhabit floor 3, so the ladder never contained the real chain).
                    r0_eff, r0_note = self._anchored_r0(sl, int(r0))
                    if r0_note:
                        decisions.append(r0_note)
                    self._open_focus(
                        sl, session_idx, latest_profile, opened_by="llm",
                        relay_r0_floor=r0_eff,
                    )
                    active.add(sl)
                    decisions.append(f"opened_relay({sl} @ R0 spawn_floor={r0_eff})")
                    self.last_relay_open = (
                        f"{sl} opened as SPAWN-ANNEAL RELAY, R0 spawn_floor={r0_eff} "
                        f"{'(habitat-anchored)' if r0_note else '(LLM-proposed)'}: siege levels "
                        f"for it must spawn at the current rung's "
                        f"floor; rung graduates on trained >={self.th.rung_graduate_sr:.0f}% x"
                        f"{self.th.rung_graduate_consecutive} fresh readings"
                    )
                    continue
            # v6fix10.1 hazard-3a: a candidate with NO failure forensics yet may not open blind —
            # the door gate below would silently wave it through (fx=None -> no substitution), which
            # is exactly the kobold-at-first-open shape ① was built for (a first-open wall was never
            # in chain_targets, so it NEVER has forensics). Park it in pending_track with the chain
            # the proposal itself supplies; chain_targets() feeds it to the ChainOrderLog, so next
            # session the gate can actually rule. Design doc line: "暂无 fail_hist → 先入链跟踪一个
            # session 再准入" — documented in fix10, implemented here.
            if self._call_forensics_provided and sl not in self._call_forensics:
                links = [
                    str(l.get("skill")).lower()
                    for l in (pf.get("prereq_tree") or [])
                    if isinstance(l, dict) and isinstance(l.get("skill"), str)
                ]
                self._nb.setdefault("pending_track", {})[sl] = {
                    "session": session_idx, "links": links,
                }
                decisions.append(
                    f"admission_deferred({sl}: no failure forensics yet — chain-tracking one "
                    f"session before the door gate can rule"
                    f"{'' if links else '; NEEDS a prereq_tree to track, keep proposing it WITH one'})"
                )
                continue
            # v6fix10 ① DOOR GATE: a wall whose failures' top missing link is still CLOSED
            # (< door_min_sr) may not be attacked through the door — the door itself opens as the
            # focus (gateway_for=<wall>), the wall stays in the queue. Attack the door, not
            # through it (fix9: kobold drilled 20 sessions behind enter_dungeon at 0%).
            door = self._door_substitute(sl, latest_profile)
            if door and door not in active and door not in watching \
                    and self._is_valid_focus(door, latest_profile):
                self._open_focus(
                    sl=door, session_idx=session_idx, latest_profile=latest_profile,
                    opened_by="llm", gateway_for=sl,
                )
                active.add(door)
                decisions.append(
                    f"door_substituted({sl}->{door}: {door} is the failures' top missing link at "
                    f"SR {latest_profile.get(door)} < {self.th.door_min_sr} — attacking the door, "
                    f"not through it)"
                )
                self.last_door_sub = (
                    f"{sl} -> {door} (top missing link below {self.th.door_min_sr:.0f}% held-out; "
                    f"the door opens as the focus, gateway_for={sl})"
                )
                continue
            self._open_focus(sl, session_idx, latest_profile, opened_by="llm")
            active.add(sl)
            decisions.append(f"opened({sl})")
        # v7fix3 P1/P0: refresh the journal mirror every reconcile (stale entries cleared — the
        # section only shows LAST session's refusals, matching what the modeler needs to correct).
        self._nb["tier_locked_last"] = tier_locked_now
        self.last_focus_decision = "; ".join(decisions) if decisions else "no_valid_focus"

    def _open_focus(
        self, sl: str, session_idx: int, latest_profile: dict[str, float], opened_by: str,
        gateway_for: str | None = None, relay_r0_floor: int | None = None,
    ) -> None:
        """Append a fresh focus (shared by the LLM-proposal path and the ② auto-open path). A wall
        re-sieged out of maintenance leaves the maintenance registry (its siege privileges resume).
        v6fix10 ①: ``gateway_for`` marks a DOOR opened in place of a still-unreachable wall.
        v7: ``relay_r0_floor`` opens the wall as a spawn-anneal relay campaign (LLM-proposal path,
        and — v7fix5.2 — the access_auto path for enter_* roots, whose R0 comes from _anchored_r0's
        habitat facts, not teacher knowledge; the ranked-walls ② auto-open still never relays)."""
        (self._nb.get("maintenance") or {}).pop(sl, None)
        # v6fix10.1 hazard-3a: an admitted wall leaves the pending-track waiting room (its chain
        # tracking continues via the focus's own chain_targets entry). A door-substituted wall
        # (gateway_for) deliberately STAYS pending — its tracking must continue through the
        # gateway siege so the gate has fresh forensics when the wall returns to the menu.
        (self._nb.get("pending_track") or {}).pop(sl, None)
        foc = _empty_focus(sl, session_idx, latest_profile.get(sl), opened_by=opened_by)
        if gateway_for:
            foc["gateway_for"] = str(gateway_for).lower()
        if relay_r0_floor is not None and int(relay_r0_floor) >= 1:
            foc["relay"] = _new_relay(int(relay_r0_floor), session_idx)
        self._nb.setdefault("foci", []).append(foc)
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "focus_opened", "focus": sl, "opened_by": opened_by,
             **({"gateway_for": str(gateway_for).lower()} if gateway_for else {}),
             **({"relay_r0_floor": int(relay_r0_floor)} if relay_r0_floor else {})}
        )

    def _anchored_r0(self, sl: str, proposed: int) -> tuple[int, str | None]:
        """v7fix4 P1: anchor a relay's R0 to the wall's native habitat floor.

        The R0 floor is a fact about the world, not a negotiable proposal — the v7fix3 run
        anchored lizard's relay at the modeler's floor 2 while lizards inhabit floor 3 (Sewers),
        so the annealing ladder never contained the real chain. defeat_*/item walls spawn AT
        their habitat floor (R0 = the fix9 drill semantics: put the student in front of the
        thing); enter_* walls spawn ONE floor above (descending INTO the floor IS the skill —
        spawning on it could trivially fire or block the entrance achievement). Unlisted /
        shallow walls (native 0) keep the LLM's floor untouched (nothing to anchor to).

        Returns (effective_r0, correction_decision_or_None).
        """
        from auction.craftax_achievements import MAX_DUNGEON_FLOOR, native_floor_of

        # clamp keeps direct notebook callers safe even for unmapped walls (the modeler parser
        # range-checks production proposals; a test/driver calling with floor 11 must not build
        # a floor-11 ladder that jax would silently OOB-clamp — the v7fix2 P5 lesson).
        proposed = max(1, min(int(proposed), MAX_DUNGEON_FLOOR))
        if not self.th.wall_floor_anchor:
            return int(proposed), None
        nf = native_floor_of(sl)
        if nf <= 0:
            return int(proposed), None
        native_r0 = max(1, nf - 1) if sl.startswith("enter_") else nf
        if int(proposed) == native_r0:
            return native_r0, None
        return native_r0, (
            f"r0_corrected({sl}: proposed R0 floor {int(proposed)} but its habitat is floor "
            f"{nf}{' (an entrance wall spawns one floor above — the descent IS the skill)' if sl.startswith('enter_') else ''}"
            f" — R0 anchored to {native_r0})"
        )

    def _attach_relay(
        self, sl: str, r0_floor: int, session_idx: int, latest_profile: dict[str, float]
    ) -> str:
        """v7fix3 P2: upgrade an ACTIVE ordinary focus into a spawn-anneal relay campaign in place.

        Rules (each refusal returns a teaching decision string):
          - already a live relay          -> plain kept (the ask is redundant, not wrong);
          - the wall has held-out wins    -> refused: a relay is for DEPTH-blocked walls, an
            ordinary siege with wins does not need a spawn ladder;
          - relay capacity full           -> refused (relay_max campaigns);
          - r0 out of the floor range     -> refused (guards direct notebook callers; the modeler
            parser already range-checks relay_r0_floor).
        On attach: the rung machine is installed fresh and the OLD attack-form state machines are
        reset (stall / ladder / gap / ④-forcing) — they measured the natural-spawn attack form,
        which no longer exists; sr_history stays (held-out semantics are unchanged). All fix2
        relay plumbing (R6 rung contract, ③ rung-momentum discount, journal RELAY section, kit
        hint) picks the campaign up with zero extra wiring.
        """
        from auction.craftax_achievements import MAX_DUNGEON_FLOOR

        foc = next(
            (f for f in self._nb.get("foci", [])
             if str(f.get("skill", "")).lower() == sl and isinstance(f, dict)),
            None,
        )
        if foc is None:  # defensive: caller guarantees sl is active
            return f"kept({sl})"
        if self._relay_active(foc):
            return f"kept({sl}: relay already running @ floor {foc['relay'].get('spawn_floor')})"
        if not 1 <= int(r0_floor) <= MAX_DUNGEON_FLOOR:
            return (
                f"relay_attach_refused({sl}: relay_r0_floor={r0_floor} outside 1..{MAX_DUNGEON_FLOOR})"
            )
        sr = latest_profile.get(sl)
        best = foc.get("best_sr")
        tol = self.th.zero_win_max_sr
        has_wins = (sr is not None and float(sr) > tol) or (best is not None and float(best) > tol)
        if has_wins:
            return (
                f"relay_attach_refused({sl}: wall has held-out wins (SR {sr}%, best {best}%) — "
                f"a relay is for depth-blocked walls that never win; keep the ordinary siege)"
            )
        n_relays = sum(1 for f in self._nb.get("foci", []) if self._relay_active(f))
        if n_relays >= self.th.relay_max:
            return (
                f"relay_attach_refused({sl}: {n_relays}/{self.th.relay_max} relay campaign(s) "
                f"already running — a slot frees when a campaign sews or retires)"
            )
        # v7fix4 P1: anchor R0 to the wall's habitat floor (see _anchored_r0 — the v7fix3 lizard
        # attach at floor 2 vs habitat floor 3 is the exact case this line closes).
        r0_eff, r0_note = self._anchored_r0(sl, int(r0_floor))
        foc["relay"] = _new_relay(int(r0_eff), session_idx)
        foc["relay_sewn"] = False
        # The old natural-spawn attack form is gone — its progress state machines restart.
        foc["frozen_sessions"] = 0
        foc["ladder_level"] = 0
        foc["gap_sessions"] = 0
        foc["gap_forced"] = False
        foc["attrib_depth_required"] = False
        foc["resume_lock"] = 0
        self._nb.setdefault("history", []).append(
            {"session": session_idx, "event": "relay_attached", "focus": sl,
             "relay_r0_floor": int(r0_eff),
             **({"relay_r0_proposed": int(r0_floor)} if r0_note else {})}
        )
        self.last_relay_open = (
            f"{sl} UPGRADED to SPAWN-ANNEAL RELAY (attached to the running focus), R0 "
            f"spawn_floor={int(r0_eff)}{' (habitat-anchored)' if r0_note else ''}: siege levels "
            f"for it must now spawn at the current "
            f"rung's floor; rung graduates on trained >={self.th.rung_graduate_sr:.0f}% x"
            f"{self.th.rung_graduate_consecutive} fresh readings"
        )
        self._save()
        return (
            f"relay_attached({sl} @ R0 spawn_floor={int(r0_eff)}: ordinary siege converted)"
            + (f"; {r0_note}" if r0_note else "")
        )

    def _maintenance_block_reason(self, sl: str, latest_profile: dict[str, float]) -> str | None:
        """v6fix8 ①: non-None (the decision string) when ``sl`` is a maintained wall still holding
        above the re-siege collapse line — sieging it again would just re-pay the monopoly tax."""
        reg = (self._nb.get("maintenance") or {}).get(sl)
        if not isinstance(reg, dict):
            return None
        sr = latest_profile.get(sl)
        collapse_line = self.th.graduate_sr - self.th.maint_resiege_drop_pp
        if sr is None or float(sr) < collapse_line:
            return None  # truly collapsed (or unreadable) — re-siege is legal
        return (
            f"maintained({sl}: graduated s{reg.get('graduated_session')}, SR {sr}% still >= "
            f"{collapse_line}% — rehearsal holds it, no re-siege)"
        )

    # ---- v7fix4.2: deep-wall relay trigger (journal directive + K-session force) -----------------

    def _relay_succession(self) -> tuple[dict[str, int], str, str] | None:
        """v7fix4.5 P1: succession context from the most recent relay-stalled retirement that
        carries a VERIFIED attribution in RELAY_SUCCESSION_CLASSES.

        v7fix4.6 P3: ``execution_failure`` joins ``chain_unreached`` — the 2026-07-13 run's
        verified diagnosis was execution_failure key=enchant_sword (can reach, can fight, lacks
        the gear), exactly as actionable as a chain break; the class-only boundary predates
        execution_failure carrying keys. An entrance key never outranks (entrances fall as
        byproducts of any deep campaign and are excluded from promotion below).

        Returns (rank_map: chain skill -> tree position, retired_wall, key_missing_link), or None
        when no qualifying retirement exists (or the knob is off). The rank map is the retired
        wall's OWN prereq_tree order (shallow -> deep, exactly as the modeler ordered it), so the
        trigger campaigns the missing equipment/magic enablers in natural learning order before
        any fresh fight wall — and before re-opening the fight that diagnosed them."""
        if not self.th.relay_succession:
            return None
        best: tuple[int, dict[str, int], str, str] | None = None
        for sl, reg in (self._nb.get("retired") or {}).items():
            if not isinstance(reg, dict):
                continue
            if str(reg.get("last_event") or "") != "focus_retired_relay_stalled":
                continue
            fa = reg.get("failure_attribution_at_retirement") or {}
            if not (
                fa.get("verified")
                and str(fa.get("class") or "") in RELAY_SUCCESSION_CLASSES
            ):
                continue
            ls = int(reg.get("last_session", -1))
            if best is None or ls > best[0]:
                # Entrances are EXCLUDED from promotion: they fall as byproducts of any deep
                # campaign (the same rationale the base order uses to put defeat_* first) — the
                # succession is about the equipment/magic/fight enablers the diagnosis names.
                links = [
                    str(x).lower() for x in (reg.get("links_at_retirement") or [])
                    if not str(x).lower().startswith("enter_")
                ]
                rank = {s: i for i, s in enumerate(links)}
                missing = str(fa.get("key_missing_link") or "").lower()
                # v7fix4.6: an entrance key must not re-enter via the -1 override (entrances are
                # excluded from promotion two lines up — the override used to add them back).
                if missing and missing.startswith("enter_"):
                    missing = ""
                if missing:
                    rank[missing] = -1  # the diagnosed link itself outranks its chain-mates
                best = (ls, rank, str(sl).lower(), missing)
        return None if best is None else (best[1], best[2], best[3])

    def _relay_trigger_candidates(
        self, latest_profile: dict[str, float] | None, session_idx: int | None = None
    ) -> list[str]:
        """Eligible deep walls for the relay trigger, best-first (deterministic).

        Eligible = native floor >= deep_wall_relay_floor AND tier < 4, not an active focus/watch,
        not cooling down or blacklisted, and no real held-out win yet (the same evidence rule the
        autoconvert branch applies). Order: fight walls (defeat_*) first — H1 lives in combat
        walls; entrances fall as chain byproducts — then shallower native floor, then name.
        Deterministic so the journal directive and the K-session force pick the SAME wall.

        v7fix4.5 P1 succession override: when the last relay campaign retired with a VERIFIED
        chain_unreached attribution, the walls of ITS chain outrank everything (tree order) —
        the run's own diagnosis, not a fresh guess, picks the next campaign.
        """
        from auction.craftax_achievements import (
            WALL_NATIVE_FLOOR,
            native_floor_of,
            tier_of,
        )

        prof = {str(k).lower(): v for k, v in (latest_profile or {}).items()}
        active = {
            f["skill"].lower() for f in self._nb.get("foci", [])
            if isinstance(f.get("skill"), str)
        }
        watching = {str(s).lower() for s in (self._nb.get("watch") or {})}
        retired = self._nb.get("retired") or {}
        out: list[str] = []
        for skill in WALL_NATIVE_FLOOR:
            sl = str(skill).lower()
            nf = native_floor_of(sl)
            if nf < self.th.deep_wall_relay_floor or tier_of(sl) >= 4:
                continue
            if sl in active or sl in watching:
                continue
            sr = prof.get(sl)
            if sr is not None and float(sr) > self.th.zero_win_max_sr:
                continue
            reg = retired.get(sl)
            if reg:
                if self._blacklist_count(reg) >= self.th.blacklist_retirements:
                    continue
                if session_idx is not None and (
                    session_idx - int(reg.get("last_session", -10**9))
                ) < self.th.cooldown_sessions:
                    continue
            out.append(sl)
        succ = self._relay_succession()
        rank = succ[0] if succ else {}
        out.sort(key=lambda s: (
            rank.get(s, 10**6),  # v7fix4.5: succession chain first, in tree order
            not s.startswith("defeat_"), native_floor_of(s), s,
        ))
        return out

    def _relay_trigger_tick(
        self, session_idx: int, latest_profile: dict[str, float], proposed_foci: list[dict]
    ) -> None:
        """v7fix4.2 (2)+(3): advance the trigger state; force-open after K ignored decisions.

        Runs once per MATURE siege decision, after reconcile + auto-open. "Answered" = this
        session's proposal contained ANY deep wall (native >= deep_wall_relay_floor), with or
        without relay_r0_floor, whatever gate it then hit — agency respected, counter resets.
        The counter only advances on decisions where the directive was ALREADY rendered (armed
        at the previous decision), so the LLM always gets the full K chances to answer. The
        trigger disarms (counter reset) whenever the conditions stop holding (a relay is live /
        no free slot / no candidate). State persists in the notebook (resume-safe).
        """
        from auction.craftax_achievements import native_floor_of

        st = self._nb.get("relay_trigger") or {}
        prev_armed = bool(st.get("armed"))
        n_relays = sum(1 for f in self._nb.get("foci", []) if self._relay_active(f))
        candidates = self._relay_trigger_candidates(latest_profile, session_idx=session_idx)
        conditions = (
            self.th.deep_wall_autoconvert
            and self.th.relay_trigger_hint
            and n_relays < self.th.relay_max
            and self._may_open_new_focus(latest_profile, relay=True)
            and bool(candidates)
        )
        if not conditions:
            self._nb["relay_trigger"] = {"armed": False, "ignored": 0, "candidates": []}
            return
        answered = any(
            isinstance(pf.get("skill"), str)
            and native_floor_of(pf["skill"].lower()) >= self.th.deep_wall_relay_floor
            for pf in (proposed_foci or [])
        )
        ignored = 0 if (answered or not prev_armed) else int(st.get("ignored", 0)) + 1
        if ignored >= self.th.relay_trigger_force_sessions:
            # (3) K-session force: the reproducibility backstop. Bounded exception to "auto-open
            # never builds a relay" — escalation-only, the LLM had K armed decisions to answer.
            sl = candidates[0]
            r0_eff, _ = self._anchored_r0(sl, native_floor_of(sl))
            self._open_focus(
                sl, session_idx, latest_profile, opened_by="relay_trigger",
                relay_r0_floor=r0_eff,
            )
            self._nb.setdefault("history", []).append(
                {"session": session_idx, "event": "relay_forced", "focus": sl, "r0": r0_eff}
            )
            self.last_relay_open = (
                f"relay_forced({sl} @ R0 spawn_floor={r0_eff}): the ★RELAY TRIGGER directive "
                f"was ignored {ignored}/{self.th.relay_trigger_force_sessions} consecutive "
                f"decisions, so the system opened the top eligible deep wall itself — treat it "
                f"as your relay campaign from now on: supply its prereq_tree and style_note "
                f"like any relay wall"
            )
            self._nb["relay_trigger"] = {"armed": False, "ignored": 0, "candidates": []}
            return
        st_new: dict = {"armed": True, "ignored": ignored, "candidates": candidates[:6]}
        # v7fix4.5 P1: surface the succession context so the journal directive can EXPLAIN why
        # the queue leads with equipment/magic walls (the run's own verified diagnosis), instead
        # of silently reordering — the modeler must be able to argue with the evidence.
        succ = self._relay_succession()
        if succ and candidates and candidates[0] in succ[0]:
            st_new["succession_from"] = succ[1]
            st_new["succession_missing"] = succ[2]
        self._nb["relay_trigger"] = st_new

    def _auto_open_from_ranked(
        self, session_idx: int, latest_profile: dict[str, float], ranked_walls: list | None
    ) -> None:
        """v6fix8 ② MULTI-FOCUS HARD GATE: fill a free, expand-gate-satisfied focus slot from the
        modeler's ranked_walls when its own foci proposal left the slot empty.

        Candidate filter = the same gates a proposed focus passes in _reconcile_foci (scope,
        conquered-held, maintenance, cooldown/blacklist — the new-tactic check is waived since an
        auto-open has no proposal text; the modeler writes the tactic next session). Preference:
        while NO active focus is COMBAT, the first viable COMBAT candidate outranks earlier
        non-combat ones (H1 lives in combat walls; enablers are budget-capped anyway). At most ONE
        auto-open per session — deliberate, so foci ramp one wall at a time."""
        from auction.craftax_achievements import family_of

        if not ranked_walls or not self._may_open_new_focus(latest_profile):
            return
        candidates: list[str] = []
        for rw in ranked_walls:
            skill = rw.get("skill") if isinstance(rw, dict) else rw
            if isinstance(skill, str) and skill.strip():
                candidates.append(skill.lower())
        active = {f["skill"].lower() for f in self._nb.get("foci", []) if isinstance(f.get("skill"), str)}
        watching = {str(s).lower() for s in (self._nb.get("watch") or {})}
        # v6fix10 ②'-4: walls whose GATEWAY door is still genuinely closed-ish are OFF the
        # auto-open menu — otherwise the moment the door yielded the wall's own door check would
        # pass again and auto-open would re-attack through a barely-cracked door. The LLM may
        # still open it via an explicit foci proposal (agency kept; it must argue why).
        # v6fix10.1 hazard-5: the block lifts once the door is INSIDE the learnable band
        # (>= gateway_release_sr, default 20) — NOT at full door graduation (>=50x2). A door whose
        # natural plateau sits mid-band (mines: baseline s151 = 21.7%) would otherwise lock its
        # wall out of auto-open for the whole run, erasing the exact fix8 trajectory (gnome opened
        # at mines ~6%) that produced the only H1 positive. Past the release line, ③ discount +
        # ④ short-circuit + the armed P3 police the economics of a premature reopen.
        gated_walls = set()
        for f in list(self._nb.get("foci", [])) + list((self._nb.get("watch") or {}).values()):
            if not (isinstance(f, dict) and f.get("gateway_for")):
                continue
            door_sr = latest_profile.get(str(f.get("skill", "")).lower())
            if door_sr is None or float(door_sr) < self.th.gateway_release_sr:
                gated_walls.add(str(f.get("gateway_for")).lower())

        def _viable(sl: str) -> bool:
            if sl in active or sl in watching or sl in gated_walls:
                return False
            if sl in self._call_chain_incomplete:  # v6fix10 ⑦
                return False
            # v7fix3 P1: tier-4 walls are relay-only, and an auto-open can never carry an R0
            # floor (that is teacher knowledge) — so they are simply off the auto-open menu.
            if self.th.tier4_relay_only:
                from auction.craftax_achievements import tier_of

                if tier_of(sl) >= 4:
                    return False
            # v7fix4 P1: habitat-deep walls (floor 3+) are relay-only for the same reason — an
            # auto-open would recreate the lizard-s19 ordinary siege the deep lock exists to stop.
            # (The LLM path teaches the relay re-proposal; the auto path just skips to the next
            # candidate.)
            if self.th.wall_floor_anchor:
                from auction.craftax_achievements import native_floor_of

                if native_floor_of(sl) >= self.th.deep_wall_relay_floor:
                    return False
            # v7fix1 (= v6fix11 port): validity (incl. achievement-table membership) rules BEFORE
            # the pending-track park below — a hallucinated name must not even enter chain
            # tracking (the LLM-proposal path already checks validity first; this aligns the auto
            # path).
            if not self._is_valid_focus(sl, latest_profile):
                return False
            # v6fix10.1 hazard-3a: no forensics = no door-gate ruling possible — park it for chain
            # tracking instead of opening blind (ranked_walls carry no prereq_tree, so the entry
            # starts link-less; the journal asks the modeler to supply the chain).
            if self._call_forensics_provided and sl not in self._call_forensics:
                pend = self._nb.setdefault("pending_track", {})
                if sl not in pend:
                    pend[sl] = {"session": session_idx, "links": []}
                return False
            if self._is_conquered_and_held(sl, latest_profile):
                return False
            if self._maintenance_block_reason(sl, latest_profile):
                return False
            reg = (self._nb.get("retired") or {}).get(sl)
            if reg:
                if session_idx - int(reg.get("last_session", -10**9)) < self.th.cooldown_sessions:
                    return False
                if self._blacklist_count(reg) >= self.th.blacklist_retirements and not \
                        self._has_new_evidence(sl, reg, latest_profile):
                    return False
            return True

        viable = [sl for sl in dict.fromkeys(candidates) if _viable(sl)]
        if not viable:
            return
        # v6fix10 ①: a gateway focus counts as COMBAT presence when its wall is COMBAT — the door
        # IS that fight's live front (otherwise the combat-preference would stack a second combat
        # wall on top of an active combat-gateway siege).
        has_combat_focus = any(
            family_of(str(f.get("gateway_for") or f.get("skill") or "")) == "COMBAT"
            for f in self._nb.get("foci", []) if isinstance(f, dict)
        )
        pick = None
        if not has_combat_focus:
            pick = next((sl for sl in viable if family_of(sl) == "COMBAT"), None)
        if pick is None:
            pick = viable[0]
        # v6fix10 ① DOOR GATE on the auto-open path: substitute the door when it is still closed.
        door = self._door_substitute(pick, latest_profile)
        if door and door not in active and door not in watching \
                and self._is_valid_focus(door, latest_profile):
            self._open_focus(
                sl=door, session_idx=session_idx, latest_profile=latest_profile,
                opened_by="auto", gateway_for=pick,
            )
            self.last_auto_open = (
                f"auto_opened({door} as the DOOR of {pick} from ranked_walls: {pick}'s failures' "
                f"top missing link {door} sits below {self.th.door_min_sr:.0f}% held-out — "
                f"attacking the door, not through it)"
            )
            self.last_door_sub = f"{pick} -> {door} (auto-open path)"
            return
        self._open_focus(pick, session_idx, latest_profile, opened_by="auto")
        self.last_auto_open = (
            f"auto_opened({pick} from ranked_walls: expand gate satisfied, slot free, LLM proposal "
            f"left it empty{'; combat-preferred' if not has_combat_focus and family_of(pick) == 'COMBAT' else ''})"
        )

    def _attach_prereq_trees(self, latest_profile: dict[str, float], proposed_foci: list[dict]) -> None:
        """For each ACTIVE focus that the proposal supplied a prereq_tree for, install it (code flags).

        v7fix4 P1 (entrance autofill): afterwards, EVERY active focus whose wall is floor-bound gets
        the entrance achievements up to its habitat floor guaranteed in the chain — the v7fix3
        lizard chain ran 30 sessions without enter_sewers because the modeler's floor cognition was
        wrong and the missing histogram only tests self-reported links (fix9), so the one absent
        link could never surface itself. The entrances are (b)-class world structure the personas
        are already prompted with; inserting them makes the break-link forensics finally able to
        point at the true door."""
        trees = {}
        for pf in proposed_foci:
            skill = pf.get("skill")
            tree = pf.get("prereq_tree")
            if isinstance(skill, str) and isinstance(tree, list):
                trees[skill.lower()] = tree
        for foc in self._nb.get("foci", []):
            fskill = foc.get("skill")
            if not isinstance(fskill, str) or fskill.lower() not in trees:
                continue
            new_tree = []
            seen: set[str] = set()
            for item in trees[fskill.lower()]:
                if not isinstance(item, dict):
                    continue
                skill = item.get("skill")
                if not isinstance(skill, str) or not skill:
                    continue
                sl = skill.lower()
                if sl in seen or sl == fskill.lower():
                    continue  # dedupe; the focus itself is not a prereq link of itself
                seen.add(sl)
                sr = latest_profile.get(sl)
                new_tree.append({
                    "skill": sl,
                    "role": str(item.get("role", ""))[:120],
                    "state": mastery_from_sr(sr, self.th.mastered_sr, self.th.unmastered_sr),
                    "sr": sr,
                })
            if new_tree:
                foc["prereq_tree"] = new_tree
        # v7fix4 P1: entrance autofill — runs over ALL active foci (not only freshly-proposed
        # trees), so a stale chain missing its door is repaired the session the anchor lands.
        if self.th.wall_floor_anchor:
            self._autofill_entrances(latest_profile)

    def _autofill_entrances(self, latest_profile: dict[str, float]) -> None:
        """Guarantee every floor-bound focus's prereq_tree contains the entrance chain to its
        habitat (enter_dungeon .. enter_<habitat floor>), inserted in floor order at the FRONT
        (entrances are the shallowest links). Records ``chain_autofilled`` events + a journal-
        visible note; the inserted links carry role='entrance (autofilled)' so the modeler sees
        exactly what the system added and why."""
        from auction.craftax_achievements import FLOOR_ENTRANCES, native_floor_of

        notes: list[str] = []
        for foc in self._nb.get("foci", []):
            sl = str(foc.get("skill", "")).lower()
            nf = native_floor_of(sl)
            if nf < 1:
                continue
            # an entrance wall needs the doors BELOW it, not itself.
            top = nf - 1 if sl.startswith("enter_") else nf
            if top < 1:
                continue
            tree = [l for l in foc.get("prereq_tree", []) if isinstance(l, dict)]
            present = {
                str(l.get("skill", "")).lower() for l in tree if isinstance(l.get("skill"), str)
            }
            missing = [
                FLOOR_ENTRANCES[f] for f in range(1, top + 1)
                if FLOOR_ENTRANCES.get(f) and FLOOR_ENTRANCES[f] not in present
                and FLOOR_ENTRANCES[f] != sl
            ]
            if not missing:
                continue
            inserted = [
                {
                    "skill": e,
                    "role": "entrance (autofilled: habitat door)",
                    "state": mastery_from_sr(
                        latest_profile.get(e), self.th.mastered_sr, self.th.unmastered_sr
                    ),
                    "sr": latest_profile.get(e),
                }
                for e in missing
            ]
            foc["prereq_tree"] = [*inserted, *tree]
            notes.append(f"{sl}: +{','.join(missing)}")
            self._nb.setdefault("history", []).append(
                {"session": self._nb.get("last_session"), "event": "chain_autofilled",
                 "focus": sl, "added": missing}
            )
        self.last_chain_autofill = (
            f"chain_autofilled({'; '.join(notes)} — a floor-bound wall's chain must pass every "
            f"entrance to its habitat; the system inserted the missing door(s))"
            if notes else None
        )

    # ---- rendering for the modeler prompt (the previous-notebook context handed back to the LLM) --

    def _render_probe_lines(self, wall: str) -> list:
        """v7fix5.5: this relay wall's probe availability + pending + latest report (journal
        lines, facts only). The sensor list renders from PROBE_SENSORS — the same table the
        filter compiler validates against, so the offer can never drift from the gate."""
        lines: list = []
        sidx = int(self._nb.get("last_session") or 0)
        pend = self._nb.get("probe_pending") or {}
        budget = self._probe_budget_left(wall, sidx)
        if pend and str(pend.get("wall")) == wall:
            lines.append(
                f"      PROBE PENDING ({pend.get('kind')}, asked s"
                f"{pend.get('requested_session')}): runs between sessions — the report lands "
                "on your next page."
            )
        elif budget.get("diagnose", 0) > 0 or budget.get("whatif", 0) > 0:
            # v7fix5.5 emission nudge (smoke 2026-07-17: with only a passive availability line
            # the modeler used the tool 0/3 even on a stalled rung — the fix42 lesson again:
            # generic teaching is ignored, a decision-moment DIRECTIVE works). Render the ★ask
            # exactly when measuring is the right move: rung stalled AND budget unspent.
            foc = self._relay_foc(wall)
            if foc is not None and not self._relay_progressing(foc):
                # v7fix5.6 double hardening (2026-07-18: at a 6-reading entry stall the modeler
                # theorized past 4 nudged pages by citing its stale report as "FRESH evidence"):
                # (a) a report older than PROBE_STALE_SESSIONS voids the escape clause in so
                # many words; (b) at stall_patience >= 4 (2 readings from early-stop) the clause
                # is dropped entirely — a hard directive, no judgment left to the LLM.
                _r56 = foc.get("relay") or {}
                _n55 = int(_r56.get("readings_since_transition", 0))
                _pat56 = int(_r56.get("stall_patience", 0))
                _rep56 = (self._nb.get("probe_reports") or {}).get(wall)
                _age56 = (
                    sidx - int(_rep56.get("delivered_session") or 0)
                    if isinstance(_rep56, dict) else None
                )
                _stale56 = _age56 is None or _age56 > PROBE_STALE_SESSIONS
                _head56 = (
                    f"      ★MEASURE BEFORE THEORIZING: this rung has made no transition for "
                    f"{_n55} readings. Before refining the tactic AGAIN on the same evidence, "
                    f"request ONE probe THIS session (siege_update.probe_request) — diagnose "
                    f"(what actually happens in this rung's episodes) or whatif (one scaffold "
                    f"knob, paired zero-shot delta)"
                )
                if _pat56 >= 4:
                    lines.append(
                        _head56 + ". This is a DIRECTIVE — patience is nearly exhausted and "
                        "the campaign is 2 readings from retirement review; emit the "
                        "probe_request NOW."
                    )
                elif _stale56:
                    lines.append(
                        _head56 + ". Your only report is "
                        + (f"{_age56} sessions old" if _age56 is not None else "absent")
                        + " — it does NOT count as fresh evidence, and the "
                        "cite-fresh-evidence exemption does not apply."
                    )
                else:
                    lines.append(
                        _head56 + " — unless you can already cite FRESH "
                        "evidence for your next move."
                    )
            lines.append(
                f"      PROBE TOOL available for this wall (budget left this "
                f"{PROBE_BUDGET_WINDOW_FAST}-session window: diagnose {budget.get('diagnose', 0)}, "
                f"whatif {budget.get('whatif', 0)}). Sensors for filters: "
                + ", ".join(sorted(PROBE_SENSORS)) + "."
            )
            # v7fix5.7-P2' §3.5 (the E5 fix): the FEASIBLE-AXIS MENU. Every axis shows which
            # directions can still move at the current stage; a dead axis says EXHAUSTED so
            # the scientist stops re-proposing boundary interventions (E5: needs_clock was
            # re-proposed at its softest value twice). Computed from probe_variant_knobs —
            # the SAME stepper that executes interventions, so the menu can never drift.
            _menu57 = []
            for _ax57 in PROBE_AXES:
                _dirs57 = []
                for _dir57 in ("easier", "harder"):
                    _k57m, _d57m, _e57m = self.probe_variant_knobs(wall, _ax57, _dir57)
                    if _e57m is None:
                        _dirs57.append(f"{_dir57} ({_d57m})")
                _menu57.append(
                    f"{_ax57}: " + ("; ".join(_dirs57) if _dirs57 else "EXHAUSTED")
                )
            lines.append(
                "      AXIS MENU (what can still move at this stage — do not propose an "
                "EXHAUSTED axis): " + " | ".join(_menu57)
            )
        rep = (self._nb.get("probe_reports") or {}).get(wall)
        if isinstance(rep, dict):
            lines.extend(self._render_probe_report(rep))
        lines.extend(self._render_hypothesis_lines(wall))
        return lines

    def _render_hypothesis_lines(self, wall: str) -> list:
        """v7fix5.5 P2: this wall's latest hypothesis entries (<= 2) with their verdict state —
        the REFUTED feedback loop lives here (the scientist's own hypothesis, judged by the
        machine, handed back; never researcher knowledge). Facts + status only."""
        entries = [
            e for e in (self._nb.get("hypothesis_log") or [])
            if isinstance(e, dict) and e.get("wall") == wall
        ][-2:]
        bar = self.th.hypothesis_verify_delta_pp
        lines: list = []
        for e in entries:
            st = str(e.get("status", ""))
            delta = e.get("delta_pp")
            d_txt = f"{float(delta):+.1f}pp" if isinstance(delta, (int, float)) else "?"
            tail = {
                "rejected_shape": " — REJECTED (malformed block).",
                "rejected_tier1": (
                    " — REJECTED: the evidence cited no verifiable report/reading number. "
                    "File hypotheses with the report's own numbers."
                ),
                "recorded": " — recorded; verification waits for a stalled rung + a free slot.",
                "verify_scheduled": (
                    " — a paired one-knob measurement runs between sessions; verdict on a "
                    "later page."
                ),
                "verified_compiled": (
                    f" — ★VERIFIED (paired delta {d_txt} >= {bar:.0f}pp): compiled into the "
                    f"ladder as an INSERTED rung (graduating it returns to stage "
                    f"{e.get('return_stage')})."
                ),
                "refuted": (
                    f" — ★REFUTED (paired delta {d_txt} < {bar:.0f}pp): the named mechanism "
                    f"is NOT the binding constraint. Theorize differently from the same "
                    f"report, or measure something else."
                ),
                "insert_graduated": (
                    " — its compiled rung GRADUATED; the ladder is back on its regular stage."
                ),
                "insert_stalled": (
                    " — its compiled rung STALLED and was removed; the regular ladder "
                    "resumed (the machine retracts wrong hypotheses on its own)."
                ),
                "stale_context": (
                    f" — verdict (delta {d_txt}) arrived after the ladder moved; recorded "
                    f"only, nothing compiled."
                ),
                "compile_refused": (
                    f" — VERIFIED (delta {d_txt}) but not compiled "
                    f"({e.get('note', 'refused')})."
                ),
                "unverifiable": (
                    # v7fix5.7-P2' §3.5: same hardness as refuted — an unverifiable-at-boundary
                    # hypothesis is structurally falsified, not merely postponed (E5).
                    f" — ★UNVERIFIABLE ({e.get('note', 'axis at boundary')}): this axis is "
                    f"ALREADY at its easiest value at this stage — it CANNOT be the binding "
                    f"constraint. Propose a DIFFERENT axis."
                ),
                "expired": " — expired unverified.",
            }.get(st, f" — {st}.")
            lines.append(
                f"      HYPOTHESIS (s{e.get('session')}, intervention {e.get('axis')} -> "
                f"{e.get('direction')}): {e.get('hypothesis')}{tail}"
            )
        return lines

    def _render_probe_report(self, rep: dict) -> list:
        """One delivered probe report -> journal lines. Numbers only — the interpretation is the
        modeler's job (that is the whole point of the tool). STALE stamping: a report older than
        PROBE_STALE_SESSIONS at render time is background, not citable evidence."""
        def _g(v):
            return f"{v:g}" if isinstance(v, (int, float)) else str(v)

        sidx = int(self._nb.get("last_session") or 0)
        delivered = int(rep.get("delivered_session") or 0)
        head = (
            f"      PROBE REPORT ({rep.get('kind')}, measured s{delivered}"
            + (f", ckpt {rep.get('ckpt_step')}" if rep.get("ckpt_step") is not None else "")
            + ")"
        )
        if sidx - delivered > PROBE_STALE_SESSIONS:
            head += (
                f" ★STALE — training has moved > {PROBE_STALE_SESSIONS} sessions since; treat "
                "as background, do NOT cite it to justify a new request"
            )
        lines = [head + ":"]
        if rep.get("kind") == "whatif":
            lines.append(
                f"        {rep.get('step_desc')}: base zero-shot SR "
                f"{_g(rep.get('base_success_pct'))}% -> variant "
                f"{_g(rep.get('variant_success_pct'))}% (delta {_g(rep.get('delta_pp'))}pp "
                f"on {_g(rep.get('n_envs'))} paired worlds, same seeds, no training)"
            )
            return lines
        filt = rep.get("filter_used")
        lines.append(
            f"        episodes: n={_g(rep.get('n_envs'))}, success {_g(rep.get('success_pct'))}%"
            f" / died {_g(rep.get('died_pct'))}% / timeout {_g(rep.get('timeout_pct'))}%"
            + (
                f"; snapshots filtered by {filt['field']}{filt['op']}{_g(filt['value'])} "
                f"(matched {_g(rep.get('filter_matched'))})"
                if isinstance(filt, dict) else "; snapshots uniform random"
            )
            + (
                f" (your filter was invalid: {rep.get('filter_error')} -> random used)"
                if rep.get("filter_error") else ""
            )
        )
        marg = rep.get("marginals") or {}
        if marg:
            cells = []
            for f, st in sorted(marg.items()):
                if isinstance(st, dict) and "rate_pct" in st:
                    cells.append(f"{f}={_g(st['rate_pct'])}%")
                elif isinstance(st, dict):
                    cells.append(
                        f"{f} p25/med/p75={_g(st.get('p25'))}/{_g(st.get('med'))}/"
                        f"{_g(st.get('p75'))}"
                    )
            lines.append("        death-snapshot marginals: " + "; ".join(cells))
        snaps = rep.get("snapshots") or []
        if snaps:
            lines.append(f"        {len(snaps)} raw episode-end snapshots:")
            for s in snaps:
                if isinstance(s, dict):
                    lines.append(
                        "          - " + ", ".join(f"{k}={_g(v)}" for k, v in sorted(s.items()))
                    )
        return lines

    def render_for_prompt(self) -> str:
        """Compact text of the current notebook to feed back into the modeler next session.

        This is the 'read the previous page of your journal' input for the A-layer LLM update.
        """
        nb = self._nb
        foci = nb.get("foci", [])
        # v7fix4.2: an ARMED relay trigger must reach the modeler even on an otherwise empty
        # notebook — the empty state (fresh run / everything graduated) is exactly when the
        # directive matters most, and the old early-return would have swallowed it.
        _rt_armed = bool((nb.get("relay_trigger") or {}).get("armed")) \
            and bool((nb.get("relay_trigger") or {}).get("candidates"))
        if not foci and not nb.get("verified_chains") and not nb.get("retired") \
                and not nb.get("maintenance") and not nb.get("watch") \
                and not nb.get("pending_track") and not nb.get("tier_locked_last") \
                and not _rt_armed:
            return "(empty siege notebook — no active focus yet; pick the first wall this session.)"
        lines = []
        # v7fix3 P1/P0: last session's tier-locked refusals, mirrored here because the decision
        # string itself is log-only — the modeler must SEE the refusal to correct its proposal.
        if nb.get("tier_locked_last"):
            locked = ", ".join(sorted(str(s) for s in nb["tier_locked_last"]))
            lines.append(
                f"★TIER-LOCKED LAST SESSION: {locked} — refused as ordinary siege(s): these "
                f"deepest-layer walls are RELAY-ONLY. To attack one, re-propose it WITH "
                f'"relay_r0_floor": <its floor> (a spawn-anneal campaign); an ordinary '
                f"re-proposal will be refused again."
            )
        # v7fix4.2 (2): the deep-wall relay trigger directive — a decision-moment concrete
        # instruction (fix1 f2: specific beats generic), rendered while armed by the tick.
        _rt = nb.get("relay_trigger") or {}
        if _rt.get("armed") and _rt.get("candidates"):
            from auction.craftax_achievements import native_floor_of as _nf_of

            _cand = ", ".join(f"{s} (floor {_nf_of(s)})" for s in _rt["candidates"])
            _k = self.th.relay_trigger_force_sessions
            lines.append(
                f"★RELAY TRIGGER — no spawn-anneal campaign is running and a focus slot is "
                f"free. The floor-3+ walls ({_cand}) can NEVER fall to any natural-spawn "
                f"curriculum (their exposure from natural spawn is a product of per-floor "
                f"descent rates), and a relay campaign needs tens of sessions — the later it "
                f"starts, the less likely it finishes this run. PUT ONE of these walls in "
                f"your foci THIS session: an ordinary proposal auto-converts into a relay "
                f"campaign (decision: relay_converted). This directive has been ignored "
                f"{int(_rt.get('ignored', 0))}/{_k} consecutive decisions; at {_k} the "
                f"system opens the top candidate itself (decision: relay_forced)."
            )
            # v7fix4.5 P1: succession — the queue order above is the run's own verified
            # diagnosis, and the modeler is told so (evidence, not fiat).
            if _rt.get("succession_from"):
                # v7fix4.6 P3: class-neutral wording — the diagnosis may be chain_unreached
                # (the chain never reached the enabler) OR execution_failure (reached, fought,
                # lacked it); both name the same actionable missing enabler.
                _miss = _rt.get("succession_missing") or "a chain enabler"
                lines.append(
                    f"  SUCCESSION: your {_rt['succession_from']} campaign retired with a "
                    f"VERIFIED diagnosis — the fight was lost for lack of {_miss}. The "
                    f"candidates above therefore lead with that campaign's own unmastered "
                    f"equipment/magic walls, in chain order: campaign the enabler FIRST; "
                    f"re-open {_rt['succession_from']} after it is SEWN and the cooldown "
                    f"expires."
                )
        # v7fix5.5: the probe gate's last answer (accept / reject / fail receipt) — the modeler
        # must SEE the refusal reason or it will re-ask blind (same mirror rule as tier_locked).
        if nb.get("probe_receipt"):
            lines.append(f"PROBE RECEIPT: {nb['probe_receipt']}")
        # §2.6: render every active focus (up to max_focus parallel sieges).
        if foci:
            from auction.craftax_achievements import family_of

            lines.append(f"ACTIVE FOCI ({len(foci)}/{self.th.max_focus}) — the hard walls being attacked:")
            for foc in foci:
                skill_l = str(foc.get("skill", "")).lower()
                lines.append(
                    f"  * {foc.get('skill')} "
                    f"(started s{foc.get('started_session')}, best SR {foc.get('best_sr')}%, "
                    f"frozen {foc.get('frozen_sessions', 0)} session(s) [whole-tree no-progress], "
                    f"form so far: {foc.get('last_siege_type') or 'unknown'})"
                )
                # v6fix8 ②: an auto-opened focus must be announced (accountability — the modeler
                # will otherwise see a focus it never proposed and may fight it).
                if str(foc.get("opened_by", "llm")) == "auto":
                    lines.append(
                        "      (opened AUTOMATICALLY by the system from your ranked_walls when the "
                        "slot freed — adopt it: backtrack its prereq chain + write its tactic now)"
                    )
                # v7: a relay focus renders its rung ladder state (code-driven; the modeler is
                # told, never asked to manage transitions).
                if isinstance(foc.get("relay"), dict):
                    r = foc["relay"]
                    if foc.get("relay_sewn"):
                        lines.append(
                            f"      (RELAY SEWN: the spawn-anneal ladder from floor "
                            f"{r.get('r0_floor')} reached natural spawn — this is now a normal "
                            "siege focus; held-out SR is the yardstick again)"
                        )
                    else:
                        rung_i = int(r.get("r0_floor", 0)) - int(r.get("spawn_floor", 0))
                        # v7fix4.7: 1dp below 10% — a slow low-SR climb (0.5 -> 0.7 -> 1.1) must
                        # be READABLE, or the modeler can neither see nor cite it in a defence.
                        recent = [
                            (round(float(x), 1) if float(x) < 10.0 else round(float(x)))
                            for x in (r.get("rung_trained") or [])[-4:]
                        ]
                        # v7fix4.6: the scaffold sub-stage is rendered too — without it a long
                        # within-floor climb reads as "stuck on floor N for 20 sessions" with a
                        # sawtooth trained SR (each stage step resets it), and the modeler would
                        # misdiagnose healthy ladder progress as stagnation.
                        _st46 = int(r.get("sub_stage", 0) or 0)
                        _task46 = (
                            "task (a LIT arena away from the entry — this IS the target floor)"
                            if int(r.get("spawn_floor", -1)) == int(r.get("r0_floor", 0))
                            else "descent"
                        )
                        # v7fix5.3: when the descent regime is active at this stage, TELL the
                        # modeler what the world does differently (locked up-ladder, slowed
                        # survival clocks) — WORLD-RULE FACTS ONLY. No tactic dictation and no
                        # researcher-probe numbers (knowledge-leak boundary, user 2026-07-16:
                        # the tactic must come from the modeler's own evidence channels; a
                        # probe constant baked into a template is stale for every other
                        # wall/stage). The knobs are code-driven, never the modeler's to set
                        # (the v7fix3 lesson: an FM freedom not pinned by code gets levelled).
                        _sc53 = self.relay_scaffold(str(foc.get("skill", ""))) or {}
                        _knob53 = ""
                        if _sc53.get("uplock") or float(_sc53.get("needs_multiplier", 1.0)) < 1.0:
                            _knob53 = (
                                f"; DESCENT REGIME at this stage (code-set, not yours to "
                                f"change): the spawn floor's up-ladder is REMOVED (retreat "
                                f"upward is impossible in this world) and the survival "
                                f"clocks (hunger/thirst/fatigue) run at "
                                f"{float(_sc53.get('needs_multiplier', 1.0)):.1f}x, annealing "
                                f"back to 1.0x on later stages"
                            )
                        # v7fix5.5 P2: a hypothesis-compiled INSERTED rung renders as what it
                        # is — the modeler's own verified hypothesis, not a ladder split.
                        _ins_r55 = (
                            r.get("stage_insert")
                            if _st46 in (RUNG_INSERT_STAGE, RUNG_INSERT_LIGHT_STAGE)
                            and isinstance(r.get("stage_insert"), dict) else None
                        )
                        if _ins_r55 is not None:
                            _lit57 = (
                                " — now on the LIGHT-ANNEAL leg: spawn stamp removed, down "
                                "ladder still lit"
                                if _st46 == RUNG_INSERT_LIGHT_STAGE else ""
                            )
                            _st_txt = (
                                f", INSERTED rung (your verified hypothesis "
                                f"{_ins_r55.get('hypothesis_id')}: "
                                f"{_ins_r55.get('step_desc')}, measured "
                                f"{float(_ins_r55.get('delta_pp', 0.0)):+.1f}pp{_lit57}; "
                                f"graduating returns to stage {_ins_r55.get('return_stage')}"
                                f"{_knob53})"
                            )
                        else:
                            _st_txt = (
                                f", scaffold sub-stage {_st46} (the system split this floor's "
                                f"{_task46} into easier code-built sub-stages after a ~0% cliff; "
                                f"trained SR RESETS at each stage step — a sawtooth is healthy "
                                f"climbing, not regression{_knob53})" if _st46 else ""
                            )
                        lines.append(
                            f"      (SPAWN-ANNEAL RELAY, rung {rung_i} of {r.get('r0_floor')}"
                            f"{_st_txt}: "
                            f"siege levels for this wall MUST spawn at floor "
                            f"{r.get('spawn_floor')} (kit provided); recent rung trained SR "
                            f"{recent or '[]'}; the spawn moves UP one floor after trained SR "
                            f">= {self.th.rung_graduate_sr:.0f}% on {self.th.rung_graduate_consecutive} "
                            f"consecutive readings — CODE-DRIVEN, you will be told. Held-out SR "
                            f"staying 0 is EXPECTED until the ladder reaches natural spawn; do "
                            f"not abandon the wall for that reason. Refine the TACTIC for the "
                            f"current rung: descend from floor {r.get('spawn_floor')}, clear, "
                            f"reach the target.)"
                        )
                        # v7fix5.5 P0: SCAFFOLD FACTS full disclosure — every knob of THIS
                        # stage plus the diff to the NEXT stage, computed from the code's own
                        # stage table (_stage_knobs -> _scaffold_fact_clauses; never template
                        # constants). Lighting was the missing fact: radius stages torch-light
                        # the spawn and the down ladder, entry stages pre-light NOTHING — the
                        # modeler's evidence channels carry no light telemetry, so a lighting
                        # cliff was structurally undiagnosable until told the rule. WORLD-RULE
                        # FACTS only (fix53 leak boundary): no tactics, no probe numbers.
                        if _sc53:
                            _cur55 = self._scaffold_fact_clauses(_sc53)
                            _stage55 = int(_sc53.get("sub_stage", 0))
                            # v7fix5.5 P2: an inserted rung's NEXT is its return stage, not
                            # stage-1 arithmetic (which would mis-clamp the insert id).
                            _next_stage55 = (
                                int(_ins_r55.get("return_stage", 0) or 0)
                                if _ins_r55 is not None else _stage55 - 1
                            )
                            _nxt55 = self._scaffold_fact_clauses(
                                self._stage_knobs(r, _next_stage55)
                            )
                            _diff55 = "; ".join(
                                f"{k} -> {_nxt55[k]}"
                                for k in _cur55 if _nxt55[k] != _cur55[k]
                            ) or "none"
                            _nname55 = (
                                "FULL (the unscaffolded rung level)"
                                if _next_stage55 <= 0 else f"stage {_next_stage55}"
                            )
                            lines.append(
                                "      SCAFFOLD FACTS (world rules at THIS stage, all "
                                "code-set — never yours to change): "
                                + "; ".join(f"{k}: {v}" for k, v in _cur55.items()) + "."
                            )
                            lines.append(
                                f"      NEXT after graduating = {_nname55}, changes: "
                                f"{_diff55} (knobs not listed stay as above)."
                            )
                        # v7fix5.5 PROBE-AS-TOOL: availability / pending / latest report for
                        # THIS rung campaign (the probe is rung-scoped; a sewn wall reads no
                        # offer). Facts only — interpretation is the modeler's.
                        lines.extend(self._render_probe_lines(skill_l))
                        # v7fix4.7 Q2: an open defence window is a ONE-decision ask — render it
                        # with the exact numbers the code will verify a citation against.
                        if r.get("defend_pending") is not None:
                            _tail47 = [
                                (round(float(x), 1) if float(x) < 10.0 else round(float(x)))
                                for x in (r.get("rung_trained") or [])[-3:]
                            ]
                            lines.append(
                                f"      ★RELAY DEFENCE WINDOW (ONE decision): rung patience is "
                                f"exhausted, but the readings are still making new absolute "
                                f"maxima: {_tail47}. If you judge this a real slow climb, DEFEND "
                                f"keeping the campaign by CITING those exact numbers (e.g. "
                                f"'{' -> '.join(str(x) for x in _tail47)}') inside this focus's "
                                f"style_note along with your trend reading — the code verifies "
                                f"the citation against the true readings and resets patience "
                                f"(defence {int(r.get('defends_used', 0))}/"
                                f"{self.th.relay_defend_budget}). If you judge it flat noise, "
                                f"do NOT cite — the campaign retires next reading and its "
                                f"diagnosis hands to the succession."
                            )
                # v7fix3 P0: an ordinary (non-relay) focus that keeps reading zero-win is told its
                # upgrade path — v7fix2's pigman sat 6 zero readings with no exit because nothing
                # ever said "re-propose WITH relay_r0_floor" for an already-open wall.
                if not isinstance(foc.get("relay"), dict):
                    _hist = [h for h in (foc.get("sr_history") or []) if isinstance(h, (int, float))]
                    _zw_streak = 0
                    for _h in reversed(_hist):
                        if float(_h) <= self.th.zero_win_max_sr:
                            _zw_streak += 1
                        else:
                            break
                    if _zw_streak >= 3:
                        lines.append(
                            f"      ★ZERO-WIN x{_zw_streak}: this ordinary siege has read no "
                            f"held-out win for {_zw_streak} consecutive readings. If the real gap "
                            f"is FLOOR DEPTH (the wall lives deeper than the student can reach), "
                            f"re-propose THIS wall WITH relay_r0_floor=<its floor> — the system "
                            f"will UPGRADE the running focus to a spawn-anneal relay in place "
                            f"(not a duplicate proposal)."
                        )
                # v6fix10 ①: a gateway focus is announced as the DOOR of its wall.
                if foc.get("gateway_for"):
                    lines.append(
                        f"      (GATEWAY: opened as the verified DOOR of {foc.get('gateway_for')} — "
                        f"its failures' top missing link; crack this and {foc.get('gateway_for')} "
                        "returns to the queue)"
                    )
                # v6fix10.1 hazard-3c: the door gate rules only at open time; if the LATEST
                # forensics now shows a closed door for an ACTIVE focus, say so — the modeler can
                # re-plan (propose the door), and ④ + the armed P3 police the economics meanwhile.
                if not foc.get("gateway_for"):
                    fx = self._call_forensics.get(skill_l)
                    if fx:
                        locked_door = self._door_substitute(skill_l, self._last_profile or {})
                        if locked_door:
                            lines.append(
                                f"      ★DOOR-LOCKED (latest forensics): failures most miss "
                                f"{locked_door}, still < {self.th.door_min_sr:.0f}% held-out — "
                                f"this wall is being attacked through a closed door; consider "
                                f"proposing {locked_door} as the focus instead"
                            )
                # v6fix8 ⑤: enabler foci show their bounded budget so the modeler can plan.
                if family_of(skill_l) != "COMBAT":
                    lines.append(
                        f"      budget: ENABLER wall — siege decision "
                        f"{int(foc.get('siege_sessions', 0))}/{self.th.enabler_max_sessions} "
                        "(at the cap it retires; combat walls have no cap while their tree moves)"
                    )
                # v6fix8 ③: the gap gate speaks (measured overfit — stronger than any hunch).
                if foc.get("gap_forced"):
                    lines.append(
                        "      ★GAP GATE (FORCED): drills for this wall are won in their calm "
                        "sandbox but held-out is NOT following. Isolation drills are suspended — "
                        "levels attacking this wall MUST be DEPTH (full pressure) until the gap closes."
                    )
                elif int(foc.get("gap_sessions", 0)) > 0:
                    lines.append(
                        f"      gap watch: {int(foc.get('gap_sessions', 0))}/"
                        f"{self.th.gap_force_sessions} consecutive over-gap readings (trained>="
                        f"{self.th.gap_trained_min:.0f}% vs held-out lagging >="
                        f"{self.th.gap_min_pp:.0f}pp) — at the cap, DEPTH is forced for this wall."
                    )
                # v6fix7 P1a: the escalation ladder speaks — the modeler MUST see and obey its level.
                lvl = int(foc.get("ladder_level", 0))
                if lvl == 1:
                    lines.append(
                        "      ★LADDER L1: the whole attack tree has been frozen "
                        f"{foc.get('frozen_sessions')} sessions. Either switch the attack FORM "
                        "(DEPTH<->CONSOLIDATE) now, or DEFEND staying: give a concrete NEW reason and "
                        "plan in this focus's style_note (silently continuing is not allowed)."
                    )
                elif lvl == 2:
                    lines.append(
                        "      ★LADDER L2 (FORCED): still frozen after your defence. The attack FORM "
                        "is now switched by the system — recommend the OTHER form for this wall; the "
                        "proposer will be required to build it."
                    )
                elif lvl >= 3:
                    lines.append(
                        "      ★LADDER L3 (FORCED): form switch did not unfreeze it. You MUST write a "
                        "MATERIALLY DIFFERENT style_note (new tactic — a rephrase will be rejected) "
                        "stating what you abandon and what you try instead. At "
                        f"{self.th.ladder_l4} frozen sessions this focus retires with a cooldown."
                    )
                # v6fix9 P2: the committed causal claim, with its gate verdict — the modeler sees
                # what it claimed last time and whether the data backed it or threw it out.
                a = foc.get("failure_attribution") or {}
                if a:
                    if a.get("rejected"):
                        verdict = (f"your claim '{a['rejected']}' was REJECTED by the failure "
                                   f"forensics -> recorded as unknown; re-derive it from the data")
                    elif a.get("verified"):
                        verdict = "verified against failure forensics"
                    else:
                        verdict = "unverified (no forensic sample yet)"
                    key = f", key link: {a['key_missing_link']}" if a.get("key_missing_link") else ""
                    lines.append(
                        f"      failure attribution: {a.get('class', 'unknown')}{key} ({verdict})"
                    )
                if foc.get("style_note"):
                    status = str(foc.get("note_status", "active"))
                    if status == "contradicted":
                        tag = " [★CONTRADICTED by evidence — you MUST rewrite this tactic (a rephrase is rejected)]"
                    elif status == "stale":
                        tag = (f" [STALE — unsupported for >= {self.th.note_stale_sessions} sessions: "
                               "re-derive it from mechanics + the latest data, do not keep refining it]")
                    else:
                        tag = ""
                    lines.append(f"      style-so-far{tag}: {foc['style_note']}")
                tree = foc.get("prereq_tree") or []
                if tree:
                    for link in tree:
                        lines.append(
                            f"      - {link.get('skill')}: {link.get('state')} "
                            f"(SR {link.get('sr')}%) role={link.get('role') or '-'}"
                        )
                else:
                    lines.append("      (prereq chain not yet backtracked — propose it this session)")
        else:
            lines.append("ACTIVE FOCI: (none — pick the next wall this session)")

        # §2.5: success experience split into two zones so the LLM sees the H1 milestones vs the
        # background enablers, instead of one flat undifferentiated list.
        chains = nb.get("verified_chains", [])
        milestones = [c for c in chains if c.get("category") == "combat_milestone"]
        enablers = [c for c in chains if c.get("category") != "combat_milestone"]
        # v7fix5.5 L1-R2/R3 (salience diet): the style/note prose renders ONLY for entries that
        # touch the CURRENT attack (target or links intersect an active focus's skill/prereq
        # tree); everything else is a one-line record. A target that IS an active focus defers
        # its prose entirely to that focus's own style-so-far line above (same wall, ONE tactic
        # text — the newest). All notes stay in the notebook JSON — render-only, reversible.
        _foc_sk55 = {str(f.get("skill", "")).lower() for f in foci}
        _hot55 = set(_foc_sk55)
        for _f55 in foci:
            _hot55 |= {
                str(l.get("skill", "")).lower() for l in (_f55.get("prereq_tree") or [])
            }

        def _hot_chain55(c: dict) -> bool:
            return str(c.get("target", "")).lower() in _hot55 or any(
                str(l).lower() in _hot55 for l in (c.get("links") or [])
            )

        if milestones:
            lines.append(
                f"COMBAT MILESTONE EXPERIENCE ({len(milestones)}) — YOUR self-style, the transferable "
                f"wins to build on. ★Only entries marked VERIFIED are conquered ground that may be "
                f"legally compressed as a base for deeper walls; PROGRESS entries are NOT yet won:"
            )
            for c in milestones[-6:]:
                tag = "VERIFIED" if c.get("status") == "verified" else "PROGRESS"
                lines.append(
                    f"    - [{tag}] {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note") and _hot_chain55(c) \
                        and str(c.get("target", "")).lower() not in _foc_sk55:
                    lines.append(f"        style: {c['style_note']}")
        if enablers:
            lines.append(
                f"ENABLER GROUND ({len(enablers)}) — gear/prereqs experience (scaffolding, NOT "
                f"self-style). ★Only VERIFIED entries are held ground; PROGRESS is still contested:"
            )
            for c in enablers[-6:]:
                tag = "VERIFIED" if c.get("status") == "verified" else "PROGRESS"
                lines.append(
                    f"    - [{tag}] {c.get('target')} via [{', '.join(c.get('links', []))}] "
                    f"(SR {c.get('last_recorded_sr')}%, s{c.get('last_recorded_session')})"
                )
                if c.get("style_note") and _hot_chain55(c) \
                        and str(c.get("target", "")).lower() not in _foc_sk55:
                    lines.append(f"        note: {c['style_note']}")
        if nb.get("protected_set"):
            lines.append(f"PROTECTED (rehearsal) skills: {', '.join(nb['protected_set'])}")
        # v6fix7 P1a: retirement history made READABLE — the fix4 bug was retire→same-session-reopen
        # because the modeler never saw its own retirements or why they failed.
        retired = nb.get("retired") or {}
        if retired:
            # v7fix5.5 L1-R1 (salience diet): the reopen rule and the relay exemption are taught
            # ONCE in the section header (they were 3 lines of identical prose PER wall — 37% of
            # the render was this block); the exemption sentence renders only when a rendered
            # wall actually qualifies (ordinary-siege retirement — v7fix3.1 rule unchanged, and
            # the fix3 tests pin its absence when only relay-stalled walls are listed). Per wall:
            # the LATEST failed note in full, older notes one line each. Full notes stay in the
            # notebook JSON — render-only compression, reversible.
            shown = list(retired.items())[-4:]
            _exempt_any55 = any(
                reg.get("last_event") != "focus_retired_relay_stalled" for _, reg in shown
            )
            lines.append(
                "RETIRED WALLS (failed sieges — do NOT reopen with the same tactic. Reopening "
                "requires the cooldown passed AND a tactic genuinely different from the archived "
                "ones below."
                + (
                    ' Walls marked [relay-exempt]: if depth-blocked, re-proposing WITH '
                    '"relay_r0_floor" as a spawn-anneal relay is exempt from the cooldown — the '
                    "ladder is a different attack form; a genuinely new tactic is still required."
                    if _exempt_any55 else ""
                )
                + "):"
            )
            for skill, reg in shown:
                _fa55 = reg.get("failure_attribution_at_retirement") or {}
                _diag55 = (
                    f", diagnosis: {_fa55.get('class')}"
                    f"{' (verified)' if _fa55.get('verified') else ''}"
                    if _fa55.get("class") else ""
                )
                _tag55 = (
                    " [relay-exempt]"
                    if reg.get("last_event") != "focus_retired_relay_stalled" else ""
                )
                lines.append(
                    f"    - {skill}: retired {reg.get('count')}x, last s{reg.get('last_session')}, "
                    f"best SR reached {reg.get('sr_at_retirement')}%{_diag55}{_tag55}"
                )
                notes = [str(n) for n in (reg.get("failed_notes") or [])]
                for i, note in enumerate(notes[:-1], start=1):
                    _head55 = note.split(". ")[0]
                    if len(_head55) > 80:
                        _head55 = _head55[:80]
                    if len(_head55) < len(note):
                        _head55 += "…"
                    lines.append(f"        earlier failed tactic {i}: {_head55}")
                if notes:
                    lines.append(f"        failed tactic (latest): {notes[-1]}")
        # v6fix8 ①: graduated walls made READABLE — the modeler must not re-propose them as foci
        # (the code refuses anyway; showing why prevents it burning its proposal on a refused wall).
        # v6fix10 ②: the WATCH registry — yielded to natural momentum, siege privileges withdrawn.
        watch = nb.get("watch") or {}
        if watch:
            lines.append(
                "WATCHING (siege privileges withdrawn; do NOT propose these as foci — the system "
                "resumes each automatically. Two kinds: yielded-to-momentum walls resume if their "
                "own climb stalls; PARKED frontier-starved walls hibernate with their FULL campaign "
                "state — ladder, notes, attribution — and resume the moment their named frontier "
                "link moves. A park is NOT a retirement or a failed tactic: no cooldown, no "
                "blacklist strike; siege the frontier, not the parked wall):"
            )
            for wskill, w in watch.items():
                hist = [h for h in (w.get("sr_history") or []) if isinstance(h, (int, float))]
                tail = "->".join(f"{h:.0f}" for h in hist[-3:]) if hist else "?"
                gw = f"; gateway for {w.get('gateway_for')}" if w.get("gateway_for") else ""
                # v7fix5.2 P0: a frontier-starved park renders its wake-up condition — the modeler
                # must read this as "campaign hibernating behind <frontier>", not as a dead wall.
                if w.get("park_event"):
                    _pf = w.get("park_frontier")
                    _psr = w.get("park_frontier_sr")
                    lines.append(
                        f"  * {wskill} (PARKED frontier-starved since s{w.get('watch_since')}: "
                        f"wakes when {_pf} rises >= "
                        f"{'?' if _psr is None else f'{float(_psr):.0f}'}%+3pp"
                        f"{gw}; campaign state preserved)"
                    )
                else:
                    lines.append(
                        f"  * {wskill} (recent SR {tail}%{gw}, watching since "
                        f"s{w.get('watch_since')})"
                    )

        # v6fix10.1 hazard-3a: the admission waiting room made READABLE — the modeler must know a
        # wall is deferred for tracking (and that a link-less entry needs a prereq_tree to proceed).
        pending = nb.get("pending_track") or {}
        if pending:
            lines.append(
                "PENDING ADMISSION (no failure forensics yet — being chain-tracked before the door "
                "gate can rule; keep proposing each WITH its prereq_tree):"
            )
            for pskill, p in list(pending.items())[-4:]:
                has_links = bool(isinstance(p, dict) and p.get("links"))
                lines.append(
                    f"  * {pskill} (waiting since s{p.get('session') if isinstance(p, dict) else '?'}"
                    f"{'' if has_links else '; NO prereq_tree yet — it CANNOT be tracked or admitted until you propose it with one'})"
                )

        maintenance = nb.get("maintenance") or {}
        if maintenance:
            lines.append(
                "GRADUATED WALLS (maintenance — the siege pushed them over the hump; they climb on "
                "their own now, rehearsal guards them. Do NOT propose them as foci; pick the NEXT "
                "wall. Re-siege unlocks only if one truly collapses (SR back below "
                f"{self.th.graduate_sr - self.th.maint_resiege_drop_pp:.0f}%)):"
            )
            for skill, reg in list(maintenance.items())[-4:]:
                lines.append(
                    f"    - {skill}: graduated s{reg.get('graduated_session')} at "
                    f"{reg.get('sr_at_graduation')}% held-out"
                )
        return "\n".join(lines)
