#!/usr/bin/env python
"""P0-4 entrypoint: direction two's TWO REAL WINDOWS execution (E2).

Target state: TWO_REAL_WINDOWS_READY_FOR_AUDIT. The directive contract:

    Window k   : six-role board -> hypotheses + prediction signatures ->
                 AxisDirectives -> REAL EnvCoder (unique template, bounded
                 repair) -> REAL Student/Reference probe -> feedback_k
                 frozen (same-window application stays forbidden);
    Window k+1 : six-role board reads feedback_k ONLY (exactly one window
                 lag) -> exactly one verdict per active hypothesis
                 (SUPPORTED / REFUTED / INCONCLUSIVE) -> plan_{k+1} -> real
                 candidate probe -> criterion-wise selection -> 12 dynamic
                 + 4 anchors -> EXACTLY ONE optimizer update -> checkpoint
                 save/load round-trip.

Fail-closed asset gate (this round's ONLY reachable outcome):

* no real LLM transport exists in this worktree (credentials live only
  inside an injected closure; direction two holds none) ->
  REAL_MODE_BLOCKED_NO_LLM_BACKEND;
* the five shared runtime assets (StudentAdapter / ReferenceAdapter /
  CandidateProbeRunner / frozen anchor manifest / full-state training
  contract) are consume-only and ALL ABSENT here ->
  BLOCKED_WAITING_SHARED_RUNTIME with the full missing list;
* jax/craftax runtime modules absent locally -> LOCAL_RUNTIME_MODULE_MISSING.

Any blocker -> print the complete blocker list, exit 1. The entrypoint NEVER
falls back to the symbolic probe / mock backend and claims to be real
(NO_SILENT_FALLBACK); it never flips a REAL_* constant, never amends frozen
evidence, and never starts the long run.

P0-6 (REQUEST_CONTROL): the production path reuses the controller's existing
halt with NO bypass — if a window's board requests human control the loop
halts immediately after phase B: EnvCoder is not called, no probe runs, no
training step executes, the window does not advance, and a
HumanDecisionArtifact is written. This entrypoint only consumes the public
summary surface of that halt (``request_control_stopped`` /
``human_decision_artifact``) and reports it; there is no continuation path
around it.

Honesty note on the L1 static legality labels: ``CandidateEnvironment``
defaults carry ``legality_hint='MOCK_ONLY ...'`` and ``real_adapter_status=
'BLOCKED_NO_LOCAL_CRAFTAX'`` — the frozen scaffold labels required by the
funnel's L1 check. The production path does NOT silently re-stamp them; the
truthfulness of a real probe comes from the shared runner being
``real_simulator=True`` and from every feedback record binding the full
``RealProbeProvenance`` (runner id, identities, seed bank, transitions).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Dict, List, Optional, Tuple

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import persistence
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics
from d052.feedback_llm_ued.llm_backend import RealBackendAdapter
from d052.feedback_llm_ued.real_call_journal import RealCallJournal
from d052.feedback_llm_ued.real_env_coder import execute_real_env_coder
from d052.feedback_llm_ued.real_probe_feedback import (
    RealProbeFeedbackRunner,
    build_real_feedback_record,
)
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
    RuntimeAuthorizationBlocked,
    assert_real_mode_servicable,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    SharedRuntimeBundle,
    resolve_shared_runtime,
)

#: the shared simulator runtime modules a REAL window must execute against
REQUIRED_LOCAL_MODULES = ("jax", "craftax")

#: two windows exactly: window k (feedback_k frozen) + window k+1
#: (verdicts + plan_{k+1} + one optimizer update)
TWO_WINDOW_HORIZON = 2


class RealTwoWindowBlocked(RuntimeError):
    """The real two-window run cannot be serviced — fail closed."""


# ---------------------------------------------------------------------------
# blocker discovery (consume-only: direction two loads NOTHING itself)
# ---------------------------------------------------------------------------
def discover_blockers(*, bundle: SharedRuntimeBundle,
                      llm_transport: Optional[object],
                      backend_id: str, model_id: str,
                      student_init_contract: Optional[object] = None
                      ) -> List[Tuple[str, str]]:
    """Every reason the REAL two-window run cannot start, fail-closed codes
    first. An empty list is the ONLY condition under which execution may be
    attempted."""
    blockers: List[Tuple[str, str]] = []
    if student_init_contract is None:
        blockers.append((
            "STUDENT_INIT_CONTRACT_NOT_INJECTED",
            "the shared StudentInitContract object must be injected "
            "explicitly by the production launcher (consume-only: "
            "direction two loads nothing itself)"))
    if llm_transport is None:
        blockers.append((
            C.REAL_MODE_BLOCKED_NO_LLM_BACKEND,
            "no real LLM transport closure is injected in this worktree; "
            "credentials may only live inside such a closure and direction "
            "two holds none — falling back to the mock backend and claiming "
            "real is forbidden"))
    for asset in bundle.missing_assets():
        blockers.append((
            C.BLOCKED_WAITING_SHARED_RUNTIME,
            f"missing shared asset: {asset}"))
    for module_name in REQUIRED_LOCAL_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            blockers.append((
                "LOCAL_RUNTIME_MODULE_MISSING",
                f"cannot import {module_name!r} in this environment; a "
                "real Student/Reference probe requires the shared "
                "simulator runtime"))
    if not backend_id or not model_id:
        blockers.append((
            "REAL_BACKEND_IDENTITY_UNDECLARED",
            "the real backend/model identity must be declared explicitly "
            "for a real run (audit field; never derived silently)"))
    return blockers


# ---------------------------------------------------------------------------
# production-path feedback record builder (P0-3 binding)
# ---------------------------------------------------------------------------
def _expected_signature(controller, cand) -> Dict[str, float]:
    """Merged hypothesis prediction signature (same merge rule as the
    controller's own symbolic path — least-fixed wins per key)."""
    merged: Dict[str, float] = {}
    for hid in sorted(cand.distinguishes_hypothesis_ids):
        try:
            hyp = controller.ledger.get(hid)
        except KeyError:
            continue
        for key, value in hyp.predicted_signature.items():
            merged.setdefault(key, float(value))
    return merged


def build_window_real_feedback(*, window: int, plan, candidates, batch,
                               controller, probe_adapter,
                               student_binding, reference_binding):
    """One provenance-bound SimulatorFeedbackRecord per stage-1 probed
    candidate (the same staging population as the symbolic path), carrying
    the deepest observed metrics and the FULL provenance binding."""
    cand_by_id = {c.candidate_id: c for c in candidates}
    stage2_by_id = {r["candidate_id"]: r for r in batch.stage2_results}
    known_hypothesis_ids = [h.hypothesis_id
                            for h in controller.ledger.all()]
    records = []
    for stage1 in batch.stage1_results:
        cid = stage1["candidate_id"]
        cand = cand_by_id[cid]
        deepest = (stage2_by_id[cid]["metrics"]
                   if cid in stage2_by_id else stage1["metrics"])
        metrics = ProbeMetrics(**deepest)
        stage_name = "full" if cid in stage2_by_id else "fast"
        #: evidence trail recorded by the adapter at probe time
        trail = probe_adapter.probe_evidence.get(cid, [])
        evidence = trail[-1] if trail else {}
        predicted = _expected_signature(controller, cand)
        reference_stats = dict(
            episode_success_rate=metrics.reference_success_rate,
            mean_progress=metrics.reference_mean_progress,
            behavior_activation_rate=metrics.reference_behavior_activation)
        record, _provenance = build_real_feedback_record(
            feedback_id=f"fb-w{window:02d}-{cid}",
            candidate=cand,
            source_window=window,
            source_plan_id=plan.plan_id,
            known_hypothesis_ids=known_hypothesis_ids,
            predicted_signature=predicted,
            stage_metrics=ProbeMetrics(stage=stage_name,
                                       **{k: v for k, v
                                          in metrics.model_dump().items()
                                          if k != "stage"}),
            reference_stats=reference_stats,
            student_binding=student_binding,
            reference_binding=reference_binding,
            runner_id=probe_adapter.runner_id,
            seed_bank=evidence.get("seed_bank", []),
            ci_sample_count=int(evidence.get("ci_sample_count", 0)),
            student_checkpoint_hash=evidence.get(
                "student_checkpoint_hash", ""),
            reference_checkpoint_hash=evidence.get(
                "reference_checkpoint_hash", ""),
            expected_observed_match="")
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# the (this-round unreachable) real two-window execution
# ---------------------------------------------------------------------------
def run_two_real_windows(*, bundle: SharedRuntimeBundle,
                         llm_transport, backend_id: str, model_id: str,
                         state_path: str,
                         student_init_contract=None) -> dict:
    """Execute windows k and k+1 against the real shared runtime.

    Preconditions (all fail closed BEFORE any LLM call): complete runtime
    grants, complete shared bundle, a real transport, and the shared
    StudentInitContract object injected explicitly (the bundle only carries
    the RESOLVED identity binding; direction two never re-derives identity
    fields by hand). This round cannot satisfy them — discover_blockers()
    refuses first.
    """
    if student_init_contract is None:
        raise RealTwoWindowBlocked(
            "STUDENT_INIT_CONTRACT_NOT_INJECTED: the production launcher "
            "must inject the shared StudentInitContract object explicitly")
    authorization = RealRuntimeAuthorization(
        real_llm_backend=True, real_envcoder=True, real_probe=True,
        real_training=True)
    assert_real_mode_servicable(
        authorization=authorization, llm_transport=llm_transport,
        missing_assets=bundle.missing_assets())
    bundle = resolve_shared_runtime(bundle)

    gate = FeedbackLaunchGate(EXECUTION_MODE_REAL,
                              runtime_grants=authorization)
    journal = RealCallJournal()
    backend = RealBackendAdapter(
        llm_transport, backend_id=backend_id, model_id=model_id,
        authorized=True, journal=journal)
    gate.assert_backend_allowed(backend.kind)
    probe_adapter = RealProbeFeedbackRunner(
        shared_runner=bundle.probe_runner.runner, gate=gate,
        student_identity_hash=bundle.student.binding.identity_hash)

    def real_env_coder(*, window, plan_id, directives, sequence):
        #: the window's 7th LLM-family call under the unique template,
        #: journaled, with bounded repair and NO symbolic fallback
        return execute_real_env_coder(
            window=window, plan_id=plan_id, directives=directives,
            backend=backend, authorization=authorization,
            sequence=sequence, journal=journal)

    def probe_feedback_builder(*, window, plan, candidates, batch,
                               controller):
        return build_window_real_feedback(
            window=window, plan=plan, candidates=candidates, batch=batch,
            controller=controller, probe_adapter=probe_adapter,
            student_binding=bundle.student.binding,
            reference_binding=bundle.reference.binding)

    controller = FeedbackUEDController(
        C.MODE_NORMAL_FEEDBACK, backend=backend, probe_runner=probe_adapter,
        anchor_manifest=bundle.anchor_manifest.manifest,
        runtime_authorization=authorization,
        #: identity must be PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 or the
        #: resolve_student_binding ladder refuses
        student_init_contract=student_init_contract,
        training_contract=bundle.training.contract,
        real_env_coder_callable=real_env_coder,
        probe_feedback_builder=probe_feedback_builder)

    summary = controller.run(max_windows=TWO_WINDOW_HORIZON)

    # P0-6 re-verification (read-only): a REQUEST_CONTROL stop produces the
    # HumanDecisionArtifact and NOTHING else — this entrypoint reports it
    # and stops; there is no continuation path around the artifact.
    report = dict(
        target="TWO_REAL_WINDOWS_REAL_EXECUTION",
        execution_mode=EXECUTION_MODE_REAL,
        request_control_stopped=summary.request_control_stopped,
        stopped_window=summary.stopped_window,
        human_decision_artifact=summary.human_decision_artifact,
        n_windows=summary.n_windows,
        n_llm_calls=summary.n_llm_calls,
        real_llm_calls=backend.usage.real_calls,
        mock_llm_calls=backend.usage.mock_calls,
        total_simulator_transitions=summary.total_simulator_transitions,
        training=[dict(status=t.status,
                       transitions=t.student_training_transitions,
                       reason=t.reason) for t in controller.training_log],
        journal_entries=len(journal.entries),
        windows=summary.windows)
    if summary.request_control_stopped:
        report["outcome"] = ("REQUEST_CONTROL_STOPPED: the board requested "
                             "human control; no execution batch was "
                             "produced and nothing else is applied "
                             "autonomously")
        return report

    #: persist the two-window state (hash-chained snapshot) for audit
    persistence.save_controller(controller, state_path)
    report["state_path"] = state_path
    report["outcome"] = "TWO_REAL_WINDOWS_EXECUTED"
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction two: two REAL windows (fail-closed asset "
                    "gate; this round always blocks)")
    parser.add_argument("--check-only", action="store_true",
                        help="run the asset gate and report; never attempt "
                             "execution")
    parser.add_argument("--backend-id", default="",
                        help="declared real LLM backend id (audit field)")
    parser.add_argument("--model-id", default="",
                        help="declared real LLM model id (audit field)")
    parser.add_argument("--transport", default="",
                        help="dotted path to an injected real transport "
                             "closure (empty = none; direction two holds "
                             "no credentials)")
    parser.add_argument("--state-path",
                        default="reports/feedback_llm_ued/"
                                "real_two_window_state.json",
                        help="where to persist the two-window controller "
                             "state on success")
    return parser.parse_args(argv)


def _load_transport(dotted: str):
    """Explicit injection only — never guess credentials, never read env
    secrets. An empty path means NO transport (this round's reality)."""
    if not dotted:
        return None
    module_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    transport = getattr(module, attr)
    if not callable(transport):
        raise RealTwoWindowBlocked(
            f"REAL_LLM_TRANSPORT_NOT_CALLABLE: {dotted!r}")
    return transport


def main(argv=None) -> int:
    args = parse_args(argv)
    #: consume-only posture: direction two loads no shared asset itself —
    #: the bundle starts with every slot EMPTY (BLOCKED_WAITING_SHARED_
    #: RUNTIME), and only an explicit owner-side injection can bind a slot.
    #: The StudentInitContract object is likewise absent here; a future
    #: production launcher injects it together with the bound bundle.
    bundle = SharedRuntimeBundle()
    student_init_contract = None
    transport = _load_transport(args.transport)
    blockers = discover_blockers(
        bundle=bundle, llm_transport=transport,
        backend_id=args.backend_id, model_id=args.model_id,
        student_init_contract=student_init_contract)
    report = dict(
        entrypoint="scripts/run_e2_real_two_window.py",
        target="TWO_REAL_WINDOWS_READY_FOR_AUDIT",
        execution_mode="REAL",
        mode=C.MODE_NORMAL_FEEDBACK,
        window_horizon=TWO_WINDOW_HORIZON,
        real_capability_flags={
            name: bool(getattr(C, name))
            for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS},
        e2_pilot_authorized=C.E2_PILOT_AUTHORIZED,
        shared_runtime_status=bundle.status_report(),
        blockers=[dict(code=code, detail=detail)
                  for code, detail in blockers])
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                     default=str))
    if blockers:
        print(f"\nREAL TWO-WINDOW RUN BLOCKED ({len(blockers)} blocker(s)); "
              "no fallback exists and none will be invented. Codes: "
              + ", ".join(sorted({code for code, _ in blockers})),
              file=sys.stderr)
        return 1
    if args.check_only:
        print("\nasset gate passed; --check-only stops before execution")
        return 0
    try:
        outcome = run_two_real_windows(
            bundle=bundle, llm_transport=transport,
            backend_id=args.backend_id, model_id=args.model_id,
            state_path=args.state_path,
            student_init_contract=student_init_contract)
    except (RuntimeAuthorizationBlocked, RealTwoWindowBlocked) as exc:
        print(f"\nREAL TWO-WINDOW RUN BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
