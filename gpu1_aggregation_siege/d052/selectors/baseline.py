"""Baseline rungs S0 / S1 / S2 (deterministic ladder).

  S0_CANONICAL_BASELINE : NO LLM roles. Pure content-order selection -- every
                          candidate has composite 0, so the (composite DESC,
                          candidate_id ASC) tie-break selects the lexicographically
                          first k candidate_ids. The reproducible floor.
  S1_THREE_ROLE         : scoring-rank over the configured scoring roles
                          (tutor/critic/explorer); composite = mean of the
                          candidate's normalized scores over those roles.
  S2_FOUR_ROLE_MODELER  : S1 composite PLUS a deterministic modeler-alignment
                          bonus (modeler_bonus carried in the signals; the Modeler
                          runs once per session and never per-candidate LLM-scoring).

Critic policy is applied via the shared machinery on every rung.
"""
from __future__ import annotations

from d052.selectors.base import (
    SelectorSignals,
    mean_role_scores,
    select_unbudgeted,
)
from d052.schemas.selector import SelectionResult, SelectorConfig


def _role_values(config: SelectorConfig) -> list:
    return [r.value for r in config.roles]


def select_s0_baseline(config: SelectorConfig,
                       signals: SelectorSignals) -> SelectionResult:
    # content order: composite 0 for all -> deterministic candidate_id ascending
    return select_unbudgeted(config, signals, lambda sig: 0.0)


def select_s1_three_role(config: SelectorConfig,
                         signals: SelectorSignals) -> SelectionResult:
    roles = _role_values(config)
    return select_unbudgeted(
        config, signals, lambda sig: mean_role_scores(sig, roles))


def select_s2_four_role(config: SelectorConfig,
                        signals: SelectorSignals) -> SelectionResult:
    roles = _role_values(config)
    return select_unbudgeted(
        config, signals,
        lambda sig: mean_role_scores(sig, roles) + sig.modeler_bonus)
