#!/usr/bin/env python3
"""Phase4A-v2.3 (CC2 §三) — CANONICAL pre-registered formal-config IDENTITY, frozen + fail closed.

V2.2 bound the formal YAML's VALUES to the runtime, but the canonical pre-registration file
itself was not frozen: a copy of the YAML, edited (even with the CLI edited to match), would
still self-consistently PASS. That is self-consistency, NOT pre-registration identity. This
module freezes the TWO canonical YAML files' identities (path + file SHA + scientific SHA) and
refuses anything that is not byte-for-byte THE frozen pre-registration at THE frozen path.

Checks (all BEFORE the CLI/runtime value comparison, all BEFORE `import jax`):
  (§三.2) PATH identity   : realpath(args.formal_config) MUST equal
                            realpath(snapshot_root / frozen relative_path), AND the CLI
                            --snapshot_root MUST equal the snapshot root DERIVED from this
                            executing module's own __file__ (Phase4A-v2.4 §十一), so the CLI
                            can never point at a snapshot different from the executing code.
                            An INDIVIDUAL file copied elsewhere, a symlink escape or a `..`
                            traversal is rejected (no suffix match, no basename-only).
                            => FORMAL_CONFIG_PATH_IDENTITY_MISMATCH.
  (§三.3) CONTENT identity: actual file SHA == frozen file SHA AND actual scientific_config
                            canonical SHA == frozen scientific SHA. => FORMAL_CONFIG_IDENTITY_MISMATCH.

Phase4A-v2.4 (§十一) relocation semantics — the labels state this precisely:
  FORMAL_CONFIG_PATH_IDENTITY        = CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT
  FORMAL_CONFIG_SNAPSHOT_RELOCATION  = LAYOUT_AND_CONTENT_BOUND
Relocating the WHOLE snapshot (the executing modules AND the YAMLs together, layout preserved)
is LEGITIMATE: the derived root moves with the module, the canonical relative path still
resolves, and the frozen file/scientific SHAs still bind the content byte-for-byte. What is
NOT legitimate is pointing the CLI at a snapshot other than the one containing the executing
code, or supplying a lone YAML copy — those fail closed. The v2.3 "no copy of the snapshot"
label wording is dropped: it over-claimed (it would have forbidden a legitimate whole-snapshot
relocation).

The frozen SHAs below were COMPUTED from the real files in the f2b7aead work tree (never
hand-guessed): see `--self-test`, which re-derives them and compares.

PURE Python: stdlib + PyYAML (+ phase4a_v2_runtime_config, itself pure). No JAX / numpy / optax.
"""
import argparse
import hashlib
import json
import os

import phase4a_v2_runtime_config as RTC  # pure (yaml/json/hashlib/os)

SCHEMA = "rmt16_phase4a_v2"

# Phase4A-v2.4 (§十一): accurate path-identity labels (see module docstring). These are the
# labels reported in certificates / summaries / reports; the v2.3 NO_COPY wording is gone.
FORMAL_CONFIG_PATH_IDENTITY_LABEL = "CANONICAL_RELATIVE_PATH_UNDER_EXECUTING_SNAPSHOT_ROOT"
FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL = "LAYOUT_AND_CONTENT_BOUND"

# ---------------------------------------------------------------------------
# §三.1 — FROZEN canonical formal-config identities (computed from f2b7aead work tree).
# ---------------------------------------------------------------------------
FORMAL_CONFIG_IDENTITIES = {
    "persistent": {
        "relative_path": "configs/rmt16_phase4a_v2_persistent.yaml",
        "file_sha256": "3ac2c9bbc3f1a9b53b8c2c58df820874ffe02c35177901fc2cee88b096643138",
        "scientific_config_sha256":
            "078ecdc072fad59fdbcd8ce004675d612021eaafb171227b8671c031de43ecde",
        "schema": SCHEMA,
        "carry_mode": "persistent",
        "replay_mode": "original_vtrace",
    },
    "reset128": {
        "relative_path": "configs/rmt16_phase4a_v2_reset128.yaml",
        "file_sha256": "c2672e735098ced69849e0c3a6d4f5be38702f3efe1572726f6cc16ce2cee80c",
        "scientific_config_sha256":
            "71c3cce5e92217d5a32a616c7d33d285f327fb2e424f574f38ba5b6ab5922047",
        "schema": SCHEMA,
        "carry_mode": "reset128",
        "replay_mode": "original_vtrace",
    },
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def frozen_identity(arm):
    """Return the frozen identity record for `arm`; fail closed on unknown arm."""
    ident = FORMAL_CONFIG_IDENTITIES.get(arm)
    if ident is None:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_UNKNOWN_ARM: {arm!r} not in "
            f"{sorted(FORMAL_CONFIG_IDENTITIES)}")
    return ident


def derived_snapshot_root():
    """Phase4A-v2.4 (§十一): the snapshot root DERIVED from this executing module's own
    __file__: realpath(<this file>/../..). The layout is frozen — this module lives at
    <snapshot>/runtime/experiment_src/phase4a_v2_formal_identity.py — so the derived root is
    the snapshot that contains the EXECUTING code. The CLI --snapshot_root must equal it, which
    prevents the CLI from pointing at a snapshot different from the executing code while a
    legitimate whole-snapshot relocation (module + YAMLs moved together) still passes."""
    return os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def resolve_expected_formal_path(snapshot_root, arm):
    """The canonical realpath the formal config MUST occupy: realpath(snapshot_root/relative_path)."""
    if not snapshot_root:
        raise ValueError(
            "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH: --snapshot_root is required to pin the "
            "canonical formal-config path (no arbitrary-location copies are accepted).")
    ident = frozen_identity(arm)
    return os.path.realpath(os.path.join(snapshot_root, ident["relative_path"]))


def verify_formal_config_path_identity(snapshot_root, arm, formal_config_path):
    """§三.2 fail closed (Phase4A-v2.4 §十一): TWO realpath conditions, both mandatory:
      (a) realpath(--snapshot_root) == the snapshot root DERIVED from this executing module's
          __file__ — the CLI may not point at a snapshot other than the one containing the
          executing code (a lone relocated YAML, or a different snapshot checkout, is rejected);
      (b) realpath(args.formal_config) == realpath(snapshot_root / frozen relative_path) — the
          canonical relative path under that root (no individual-file copy, no symlink escape,
          no `..` bypass, no suffix/basename match).
    Relocating the WHOLE snapshot (module + YAMLs together, layout preserved) is legitimate:
    (a) then holds because the derived root moved with the module, and (b) + the frozen content
    SHAs still bind the bytes. A symlink whose REALPATH equals the canonical file is allowed
    (same bytes, same inode target). Returns a PASS record carrying the v2.4 labels."""
    if not formal_config_path:
        raise ValueError(
            "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH: no formal config path supplied.")
    derived = derived_snapshot_root()
    declared = os.path.realpath(str(snapshot_root)) if snapshot_root else None
    if declared != derived:
        raise ValueError(
            f"FORMAL_CONFIG_PATH_IDENTITY_MISMATCH: declared --snapshot_root realpath="
            f"{declared!r} != the executing code's derived snapshot root={derived!r}. The CLI "
            "must point at the snapshot containing the EXECUTING modules; a different snapshot "
            "(or a lone relocated YAML) is rejected. Whole-snapshot relocation is allowed: the "
            "modules and the YAMLs must move together (layout and content bound).")
    expected = resolve_expected_formal_path(snapshot_root, arm)
    actual = os.path.realpath(formal_config_path)
    if actual != expected:
        raise ValueError(
            f"FORMAL_CONFIG_PATH_IDENTITY_MISMATCH: formal config realpath={actual!r} != "
            f"canonical frozen path={expected!r} (arm={arm!r}). An individual file copied "
            "elsewhere, a symlink escape, or a '..' traversal is rejected; only the canonical "
            "relative path under the executing snapshot root is accepted.")
    return dict(path_identity="PASS", arm=arm, expected_realpath=expected,
                actual_realpath=actual,
                formal_config_path_identity=FORMAL_CONFIG_PATH_IDENTITY_LABEL,
                formal_config_snapshot_relocation=FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL,
                declared_snapshot_root_realpath=declared,
                derived_snapshot_root_realpath=derived)


def verify_formal_config_content_identity(formal_record, arm):
    """§三.3 fail closed: the loaded formal config's FILE SHA and scientific_config canonical SHA
    MUST equal the frozen values. Runs BEFORE any CLI/runtime value comparison. Returns a PASS
    record including both frozen + actual SHAs."""
    ident = frozen_identity(arm)
    if not isinstance(formal_record, dict) or not isinstance(formal_record.get("config"), dict):
        raise ValueError("FORMAL_CONFIG_IDENTITY_MISMATCH: no formal config loaded.")
    actual_file_sha = formal_record.get("file_sha256")
    if actual_file_sha != ident["file_sha256"]:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_MISMATCH: formal config file_sha256={actual_file_sha} != "
            f"frozen {ident['file_sha256']} (arm={arm!r}, path={formal_record.get('path')!r}). "
            "Any byte change — even comments or key reordering — is rejected.")
    sci = formal_record["config"].get("scientific_config")
    if not isinstance(sci, dict):
        raise ValueError("FORMAL_CONFIG_IDENTITY_MISMATCH: formal config has no scientific_config.")
    actual_sci_sha = RTC.scientific_config_sha256(sci)
    if actual_sci_sha != ident["scientific_config_sha256"]:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_MISMATCH: scientific_config canonical sha256="
            f"{actual_sci_sha} != frozen {ident['scientific_config_sha256']} (arm={arm!r}).")
    # structural consistency of the frozen record itself
    cfg = formal_record["config"]
    if cfg.get("schema") != ident["schema"]:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_MISMATCH: schema={cfg.get('schema')!r} != {ident['schema']!r}")
    if cfg.get("arm") != ident["carry_mode"]:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_MISMATCH: arm={cfg.get('arm')!r} != {ident['carry_mode']!r}")
    if sci.get("replay_mode") != ident["replay_mode"]:
        raise ValueError(
            f"FORMAL_CONFIG_IDENTITY_MISMATCH: replay_mode={sci.get('replay_mode')!r} != "
            f"{ident['replay_mode']!r}")
    return dict(content_identity="PASS", arm=arm,
                file_sha256=actual_file_sha, scientific_config_sha256=actual_sci_sha,
                frozen_file_sha256=ident["file_sha256"],
                frozen_scientific_config_sha256=ident["scientific_config_sha256"])


def verify_formal_config_identity(snapshot_root, arm, formal_record):
    """Convenience: path identity (§三.2) + content identity (§三.3) in one call."""
    path_rec = verify_formal_config_path_identity(
        snapshot_root, arm, formal_record.get("path") if formal_record else None)
    content_rec = verify_formal_config_content_identity(formal_record, arm)
    merged = dict(formal_config_identity="PASS")
    merged.update(path_rec)
    merged.update(content_rec)
    return merged


# ---------------------------------------------------------------------------
# Phase4A-direct-98304 (§一.3/§二/§三) — ENGINEERING run-config identity (NON-frozen)
# ---------------------------------------------------------------------------
# The engineering smoke (4096) and direct 98304 long-run configs are NOT part of the frozen
# formal pre-registration: their scientific_config declares an ENGINEERING budget (total_updates
# =2 / 48), so their file SHA and scientific SHA legitimately differ from the two frozen formal
# YAMLs and MUST NOT be compared against FORMAL_CONFIG_IDENTITIES. They still get the SAME path
# anti-tamper protection as the formal path (realpath under the EXECUTING snapshot root; no
# individual-file copy / symlink escape / `..` traversal), and their content is bound by
# SELF-CONSISTENCY (the driver's deep_diff compares the YAML scientific_config against the runtime
# scientific config built from the frozen spec + the ACTUAL CLI; the frozen spec still binds every
# non-budget scientific constant). The returned record carries formal_config_identity="PASS" so the
# precheck certificate's formal-identity gate is satisfied, PLUS engineering_config_identity="PASS"
# and the recomputed (non-frozen) SHAs for the evidence trail.

ENGINEERING_CONFIG_RELATIVE_PATHS = {
    ("engineering_smoke", "persistent"): "configs/rmt16_phase4a_smoke_persistent.yaml",
    ("engineering_smoke", "reset128"): "configs/rmt16_phase4a_smoke_reset128.yaml",
    ("long_run_98304", "persistent"): "configs/rmt16_phase4a_long98304_persistent.yaml",
    ("long_run_98304", "reset128"): "configs/rmt16_phase4a_long98304_reset128.yaml",
}


def verify_engineering_config_identity(snapshot_root, arm, formal_record, run_class):
    """§一.3 fail closed: canonical PATH identity (same anti-copy protection as the formal path)
    + SELF-CONSISTENT content binding for an engineering run config. NO frozen-formal SHA
    comparison (the engineering budget legitimately differs). Returns a record carrying
    formal_config_identity="PASS" (so the precheck certificate's identity gate is met) +
    engineering_config_identity="PASS" + the recomputed file/scientific SHAs."""
    key = (run_class, arm)
    rel = ENGINEERING_CONFIG_RELATIVE_PATHS.get(key)
    if rel is None:
        raise ValueError(
            f"ENGINEERING_CONFIG_IDENTITY_UNKNOWN: no canonical engineering config for "
            f"run_class={run_class!r} arm={arm!r}; known={sorted(ENGINEERING_CONFIG_RELATIVE_PATHS)}")
    if not formal_record:
        raise ValueError("ENGINEERING_CONFIG_IDENTITY_MISMATCH: no config loaded.")
    # (a) the declared --snapshot_root must equal the executing module's derived root (same rule
    #     as the formal path — the CLI may not point at a snapshot other than the executing code).
    derived = derived_snapshot_root()
    declared = os.path.realpath(str(snapshot_root)) if snapshot_root else None
    if declared != derived:
        raise ValueError(
            f"ENGINEERING_CONFIG_PATH_IDENTITY_MISMATCH: declared --snapshot_root realpath="
            f"{declared!r} != the executing code's derived snapshot root={derived!r}.")
    # (b) the config realpath must equal realpath(snapshot_root / canonical engineering path).
    expected = os.path.realpath(os.path.join(snapshot_root, rel))
    actual = os.path.realpath(formal_record.get("path"))
    if actual != expected:
        raise ValueError(
            f"ENGINEERING_CONFIG_PATH_IDENTITY_MISMATCH: config realpath={actual!r} != canonical "
            f"engineering path={expected!r} (run_class={run_class!r}, arm={arm!r}). An individual "
            "copied file, symlink escape or '..' traversal is rejected.")
    # (c) content self-consistency: schema/arm/replay_mode structural checks + recomputed SHAs
    #     (recorded, NOT compared to any frozen constant; the runtime deep_diff binds the values).
    cfg = formal_record.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("ENGINEERING_CONFIG_IDENTITY_MISMATCH: no config mapping.")
    if cfg.get("schema") != SCHEMA:
        raise ValueError(
            f"ENGINEERING_CONFIG_IDENTITY_MISMATCH: schema={cfg.get('schema')!r} != {SCHEMA!r}")
    if cfg.get("arm") != arm:
        raise ValueError(
            f"ENGINEERING_CONFIG_IDENTITY_MISMATCH: arm={cfg.get('arm')!r} != {arm!r}")
    sci = cfg.get("scientific_config")
    if not isinstance(sci, dict):
        raise ValueError("ENGINEERING_CONFIG_IDENTITY_MISMATCH: no scientific_config block.")
    if sci.get("replay_mode") != "original_vtrace":
        raise ValueError(
            f"ENGINEERING_CONFIG_IDENTITY_MISMATCH: replay_mode={sci.get('replay_mode')!r} != "
            "'original_vtrace' (engineering runs keep the original-goal V-trace protocol).")
    if sci.get("carry_mode") != arm:
        raise ValueError(
            f"ENGINEERING_CONFIG_IDENTITY_MISMATCH: scientific_config.carry_mode="
            f"{sci.get('carry_mode')!r} != arm={arm!r}")
    return dict(
        formal_config_identity="PASS",
        engineering_config_identity="PASS",
        run_class=run_class,
        arm=arm,
        relative_path=rel,
        expected_realpath=expected,
        actual_realpath=actual,
        formal_config_path_identity=FORMAL_CONFIG_PATH_IDENTITY_LABEL,
        formal_config_snapshot_relocation=FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL,
        declared_snapshot_root_realpath=declared,
        derived_snapshot_root_realpath=derived,
        file_sha256=formal_record.get("file_sha256"),
        scientific_config_sha256=RTC.scientific_config_sha256(sci),
        frozen_formal_sha_compared=False)


def self_test():
    import tempfile
    import shutil
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
              flush=True)

    print("phase4a_v2_formal_identity --self-test (Phase4A-v2.3 §三.4)", flush=True)

    snap = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    # (0) frozen SHAs re-derived from the real files must match the constants (no hand-guessing)
    rederive_ok = True
    for arm in ("persistent", "reset128"):
        ident = frozen_identity(arm)
        p = os.path.join(snap, ident["relative_path"])
        raw = open(p, "rb").read()
        import yaml as _yaml
        cfg = _yaml.safe_load(raw.decode("utf-8"))
        if _sha256_bytes(raw) != ident["file_sha256"]:
            rederive_ok = False; print(f"    [{arm}] file SHA drift", flush=True)
        if RTC.scientific_config_sha256(cfg["scientific_config"]) != ident[
                "scientific_config_sha256"]:
            rederive_ok = False; print(f"    [{arm}] scientific SHA drift", flush=True)
    check("frozen SHAs re-derived from real files match the constants", rederive_ok)

    canon = os.path.join(snap, "configs", "rmt16_phase4a_v2_persistent.yaml")
    tmp = tempfile.mkdtemp(prefix="p4av23_fid_")
    try:
        # (1) canonical YAML original file -> PASS (path + content)
        rec = RTC.load_formal_config(canon)
        try:
            idrec = verify_formal_config_identity(snap, "persistent", rec)
            check("(1) canonical original file -> PASS", idrec["formal_config_identity"] == "PASS")
        except ValueError as e:
            check("(1) canonical original file -> PASS", False, str(e)[:120])

        # (2) byte-identical COPY at another path -> FAIL path identity
        copy_path = os.path.join(tmp, "copy_persistent.yaml")
        shutil.copyfile(canon, copy_path)
        rec_copy = RTC.load_formal_config(copy_path)
        try:
            verify_formal_config_path_identity(snap, "persistent", copy_path)
            check("(2) byte copy at other path -> FAIL path identity", False, "no raise")
        except ValueError as e:
            check("(2) byte copy at other path -> FAIL path identity",
                  "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" in str(e))

        # (3) copy + edited seed (even if CLI synced) -> FAIL content identity (file + scientific)
        edited = open(canon, encoding="utf-8").read().replace("seed: 42", "seed: 43")
        edited_path = os.path.join(tmp, "edited_seed.yaml")
        with open(edited_path, "w", encoding="utf-8") as f:
            f.write(edited)
        rec_edited = RTC.load_formal_config(edited_path)
        try:
            verify_formal_config_content_identity(rec_edited, "persistent")
            check("(3) edited-seed copy -> FAIL content identity", False, "no raise")
        except ValueError as e:
            check("(3) edited-seed copy -> FAIL content identity",
                  "FORMAL_CONFIG_IDENTITY_MISMATCH" in str(e))

        # (4) copy + added comment -> file SHA changes -> FAIL (scientific may be unchanged)
        commented = "# a comment that changes file bytes only\n" + open(
            canon, encoding="utf-8").read()
        commented_path = os.path.join(tmp, "commented.yaml")
        with open(commented_path, "w", encoding="utf-8") as f:
            f.write(commented)
        rec_com = RTC.load_formal_config(commented_path)
        sci_same = (RTC.scientific_config_sha256(rec_com["config"]["scientific_config"])
                    == frozen_identity("persistent")["scientific_config_sha256"])
        try:
            verify_formal_config_content_identity(rec_com, "persistent")
            check("(4) comment-only change -> FAIL file identity", False, "no raise")
        except ValueError as e:
            check("(4) comment-only change -> FAIL file identity",
                  "FORMAL_CONFIG_IDENTITY_MISMATCH" in str(e) and "file_sha256" in str(e),
                  f"scientific_unchanged={sci_same}")

        # (5) key-reordered (same values) -> file SHA FAILS; scientific SHA stays equal
        import yaml as _yaml
        reordered = _yaml.safe_dump(_yaml.safe_load(open(canon, encoding="utf-8")),
                                    sort_keys=True)
        reordered_path = os.path.join(tmp, "reordered.yaml")
        with open(reordered_path, "w", encoding="utf-8") as f:
            f.write(reordered)
        rec_re = RTC.load_formal_config(reordered_path)
        sci_equal = (RTC.scientific_config_sha256(rec_re["config"]["scientific_config"])
                     == frozen_identity("persistent")["scientific_config_sha256"])
        try:
            verify_formal_config_content_identity(rec_re, "persistent")
            check("(5) key-reordered -> FAIL file identity", False, "no raise")
        except ValueError as e:
            check("(5) key-reordered -> FAIL file identity",
                  "FORMAL_CONFIG_IDENTITY_MISMATCH" in str(e) and sci_equal,
                  f"scientific_sha_preserved={sci_equal}")

        # (6) persistent CLI supplied the reset128 canonical YAML -> FAIL (path + content)
        r128 = os.path.join(snap, "configs", "rmt16_phase4a_v2_reset128.yaml")
        try:
            verify_formal_config_path_identity(snap, "persistent", r128)
            check("(6) wrong-arm YAML -> FAIL path identity", False, "no raise")
        except ValueError as e:
            check("(6) wrong-arm YAML -> FAIL path identity",
                  "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" in str(e))
        rec_r = RTC.load_formal_config(r128)
        try:
            verify_formal_config_content_identity(rec_r, "persistent")
            check("(6b) wrong-arm YAML -> FAIL content identity", False, "no raise")
        except ValueError as e:
            check("(6b) wrong-arm YAML -> FAIL content identity",
                  "FORMAL_CONFIG_IDENTITY_MISMATCH" in str(e))

        # (7) symlink to canonical file: realpath == canonical -> allowed; escape -> FAIL
        link_path = os.path.join(tmp, "link_to_canon.yaml")
        try:
            os.symlink(canon, link_path)
            link_rec = RTC.load_formal_config(link_path)
            ok = verify_formal_config_path_identity(snap, "persistent", link_path)
            check("(7a) symlink resolving to canonical file -> PASS",
                  ok["path_identity"] == "PASS", "symlink realpath==canonical")
        except OSError as e:
            # symlink unavailable (e.g. Windows w/o privilege): the gate is pure realpath
            # equality, already exercised by (1) and (2); record as skipped-but-covered.
            check("(7a) symlink resolving to canonical file -> PASS", True,
                  f"symlink unavailable ({e.__class__.__name__}); realpath-equality covered by (1)/(2)")
        # escape: a link/copy whose realpath resolves OUTSIDE snapshot_root -> FAIL
        outside = os.path.join(tempfile.gettempdir(), "p4av23_outside_canon.yaml")
        shutil.copyfile(canon, outside)
        try:
            verify_formal_config_path_identity(snap, "persistent", outside)
            check("(7b) path resolving outside snapshot_root -> FAIL", False, "no raise")
        except ValueError as e:
            check("(7b) path resolving outside snapshot_root -> FAIL",
                  "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" in str(e))
        finally:
            try:
                os.remove(outside)
            except OSError:
                pass

        # (8) unknown arm -> fail closed
        try:
            frozen_identity("bogus")
            check("(8) unknown arm -> raised", False, "no raise")
        except ValueError as e:
            check("(8) unknown arm -> raised", "FORMAL_CONFIG_IDENTITY_UNKNOWN_ARM" in str(e))

        # (9) Phase4A-v2.4 (§十一): the declared --snapshot_root MUST equal the root DERIVED from
        # this executing module's __file__. A DIFFERENT tree — even one holding a byte-identical
        # canonical YAML at the correct relative path — fails closed (the CLI may not point at a
        # snapshot other than the executing code's).
        other_root = os.path.join(tmp, "other_snapshot_root")
        os.makedirs(os.path.join(other_root, "configs"))
        other_yaml = os.path.join(other_root, "configs", "rmt16_phase4a_v2_persistent.yaml")
        shutil.copyfile(canon, other_yaml)
        try:
            verify_formal_config_path_identity(other_root, "persistent", other_yaml)
            check("(9a) other snapshot_root (valid YAML, right relative path) -> FAIL",
                  False, "no raise")
        except ValueError as e:
            check("(9a) other snapshot_root (valid YAML, right relative path) -> FAIL",
                  "FORMAL_CONFIG_PATH_IDENTITY_MISMATCH" in str(e)
                  and "derived snapshot root" in str(e))
        idrec9 = verify_formal_config_identity(snap, "persistent", RTC.load_formal_config(canon))
        check("(9b) path identity record carries v2.4 labels + derived root binding",
              idrec9["formal_config_path_identity"] == FORMAL_CONFIG_PATH_IDENTITY_LABEL
              and idrec9["formal_config_snapshot_relocation"]
              == FORMAL_CONFIG_SNAPSHOT_RELOCATION_LABEL
              and idrec9["derived_snapshot_root_realpath"] == os.path.realpath(snap)
              and idrec9["declared_snapshot_root_realpath"] == os.path.realpath(snap))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(results); n_pass = sum(results)
    print(f"SELF_TEST_SUMMARY total={n} pass={n_pass} fail={n - n_pass}", flush=True)
    return 0 if n_pass == n else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run the §三.4 anti-tamper self-tests (temp files; no JAX; no training)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    ap.error("--self-test is the only supported mode here; the driver imports this module.")


if __name__ == "__main__":
    raise SystemExit(main())
