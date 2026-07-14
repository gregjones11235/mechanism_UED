"""SiegeNotebook — central SIEGE state orchestrator.

Updates all SIEGE components each session based on held-out evidence.
Never passes tier labels to LLMs, selectors, or scoring.
"""

import json, os, time
from typing import Optional


class SiegeNotebook:
    """Central orchestrator for SIEGE curriculum enrichment."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.session_count = 0

        # Lazy imports to avoid circular deps
        from dicode.siege.student_profile import StudentProfileLog
        from dicode.siege.chain_order import ChainOrderLog
        from dicode.siege.held_out import HeldOutEvaluator
        from dicode.siege.focus_quota import FocusQuota
        from dicode.siege.rehearsal import ForgettingRehearsal

        self.profile = StudentProfileLog(os.path.join(state_dir, "student_profile.json"))
        self.chain_order = ChainOrderLog(os.path.join(state_dir, "chain_order.json"))
        self.held_out = HeldOutEvaluator(os.path.join(state_dir, "held_out.json"))
        self.focus_quota = FocusQuota(state_path=os.path.join(state_dir, "focus_quota.json"))
        self.rehearsal = ForgettingRehearsal(state_path=os.path.join(state_dir, "rehearsal.json"))

        self.state_path = os.path.join(state_dir, "notebook_state.json")
        if os.path.exists(self.state_path):
            self.load()

    # ── Chain definition ──

    def define_craftax_chains(self) -> None:
        """Define prerequisite chains for Craftax achievements."""
        # Crafting progression chain
        self.chain_order.define_chain("crafting_progression", [
            "collect_wood", "craft_planks", "craft_stick",
            "craft_wooden_pickaxe", "collect_stone",
            "craft_furnace", "craft_stone_pickaxe",
            "collect_coal", "collect_iron", "smelt_iron",
            "craft_iron_pickaxe", "collect_diamond",
        ])
        # Combat progression chain
        self.chain_order.define_chain("combat_progression", [
            "craft_wooden_sword", "defeat_zombie",
            "craft_stone_sword", "defeat_skeleton",
            "craft_iron_sword", "defeat_creeper",
            "enter_dungeon", "defeat_dungeon_skeleton",
        ])

    # ── Session update ──

    def update(self, held_out_metrics: dict, global_step: int) -> dict:
        """Update all SIEGE state from new held-out evidence.

        Args:
            held_out_metrics: Dict of achievement_name -> success_rate.
            global_step: Current global environment step count.

        Returns:
            Summary dict of all SIEGE state changes.
        """
        self.session_count += 1

        # 1. Record held-out evaluation
        self.held_out.record_evaluation(self.session_count, global_step, held_out_metrics)

        # 2. Update student profile
        self.profile.update(held_out_metrics)

        # 3. Update chain order from profile
        self.chain_order.update_from_profile(self.profile)

        # 4. Detect forgetting and update rehearsal
        rehearsal_state = self.rehearsal.update(self.profile, self.session_count)

        # 5. Save all state
        self.save()

        return {
            "session": self.session_count,
            "global_step": global_step,
            "profile_summary": self.profile.summary,
            "chain_summary": self.chain_order.summary,
            "rehearsal": {
                "active": rehearsal_state["active"],
                "count": rehearsal_state["at_risk_count"],
            },
        }

    # ── Candidate metadata ──

    def get_candidate_metadata(self, task_id: str, task_achievements: list[str]) -> dict:
        """Generate SIEGE metadata for a candidate task.

        NEVER includes tier labels — only binary mastery flags and
        chain-relevance indicators.
        """
        chain_achievements = set()
        for chain_name in self.chain_order.chains:
            for link in self.chain_order.chains[chain_name]["links"]:
                chain_achievements.add(link["achievement"])

        relevant = [a for a in task_achievements if a in chain_achievements]
        mastered = [a for a in relevant if self.profile.is_mastered(a)]
        unmastered = [a for a in relevant if not self.profile.is_mastered(a)]

        # Find break link
        break_link = None
        for chain_name in self.chain_order.active_chains:
            bl = self.chain_order.get_break_link(chain_name)
            if bl and bl["achievement"] in task_achievements:
                break_link = bl["achievement"]
                break

        return {
            "siege_wall": len(relevant) > 0,
            "active_prerequisite_chain": len(unmastered) > 0,
            "mastered_links": mastered,
            "unmastered_links": unmastered,
            "current_break_link": break_link,
            "chain_complete": len(unmastered) == 0 and len(mastered) > 0,
            "recommended_form": "chain_completion" if unmastered else "consolidation",
            "expected_frontier_gain": 1.0 if break_link else 0.0,
            "forgetting_risk": any(
                self.profile.get_forgetting_risk(a) for a in relevant
            ),
            "evidence_source": "held_out_evaluation",
        }

    # ── Persistence ──

    def load(self) -> None:
        with open(self.state_path) as f:
            data = json.load(f)
        self.session_count = data.get("session_count", 0)

    def save(self) -> None:
        self.profile.save()
        self.chain_order.save()
        self.held_out.save()
        self.focus_quota.save()
        self.rehearsal.save()
        with open(self.state_path, "w") as f:
            json.dump({"session_count": self.session_count}, f, indent=2)

    @property
    def summary(self) -> dict:
        return {
            "session_count": self.session_count,
            "profile": self.profile.summary,
            "chains": self.chain_order.summary,
            "held_out": self.held_out.summary,
            "focus_quota": self.focus_quota.summary,
            "rehearsal": self.rehearsal.summary,
        }
