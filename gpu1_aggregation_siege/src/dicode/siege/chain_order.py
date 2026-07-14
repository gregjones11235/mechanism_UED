"""ChainOrderLog — prerequisite chain tracking and frontier detection.

Tracks which prerequisite links are mastered/unmastered, identifies the
current break link where the student is stuck, and measures chain-frontier
depth progression.
"""

import json, os, time
from typing import Optional


class ChainOrderLog:
    """Manages prerequisite chains for curriculum tasks.

    A chain is an ordered list of achievements that must be mastered in sequence.
    Each link has a mastery state derived from held-out evaluation.
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path
        self.chains: dict[str, dict] = {}  # chain_name -> {links, metadata}
        self.session_count = 0
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.chains = data.get("chains", {})
        self.session_count = data.get("session_count", 0)

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.state_path
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "chains": self.chains,
                "session_count": self.session_count,
            }, f, indent=2)

    def define_chain(self, name: str, links: list[str]) -> None:
        """Define a prerequisite chain. Links must be mastered in order.

        Args:
            name: Chain identifier (e.g., 'crafting_progression').
            links: Ordered list of achievement names forming the chain.
        """
        self.chains[name] = {
            "links": [
                {"achievement": ach, "mastered": False, "sessions_mastered": None}
                for ach in links
            ],
            "frontier_depth": 0,
            "complete": False,
            "created_session": self.session_count,
        }

    def update_from_profile(self, student_profile) -> None:
        """Update chain mastery state from student held-out profile."""
        self.session_count += 1
        for name, chain in self.chains.items():
            max_mastered = 0
            for i, link in enumerate(chain["links"]):
                ach = link["achievement"]
                was_mastered = link["mastered"]
                link["mastered"] = student_profile.is_mastered(ach)
                if link["mastered"] and not was_mastered:
                    link["sessions_mastered"] = self.session_count
                if link["mastered"]:
                    max_mastered = i + 1
                else:
                    break  # Stop at first unmastered link
            chain["frontier_depth"] = max_mastered
            chain["complete"] = max_mastered == len(chain["links"])

    def get_break_link(self, chain_name: str) -> Optional[dict]:
        """Get the first unmastered link in a chain."""
        chain = self.chains.get(chain_name)
        if not chain:
            return None
        for link in chain["links"]:
            if not link["mastered"]:
                return dict(link)
        return None

    def get_prerequisite_achievements(self, chain_name: str, up_to_link: int) -> list[str]:
        """Get achievements that are prerequisites for a given link."""
        chain = self.chains.get(chain_name)
        if not chain:
            return []
        return [l["achievement"] for l in chain["links"][:up_to_link]]

    @property
    def active_chains(self) -> list[str]:
        return [n for n, c in self.chains.items() if not c["complete"]]

    @property
    def complete_chains(self) -> list[str]:
        return [n for n, c in self.chains.items() if c["complete"]]

    @property
    def max_frontier_depth(self) -> int:
        if not self.chains:
            return 0
        return max(c["frontier_depth"] for c in self.chains.values())

    @property
    def summary(self) -> dict:
        return {
            "session_count": self.session_count,
            "total_chains": len(self.chains),
            "active_chains": len(self.active_chains),
            "complete_chains": len(self.complete_chains),
            "max_frontier_depth": self.max_frontier_depth,
            "chains": {
                name: {
                    "frontier_depth": c["frontier_depth"],
                    "complete": c["complete"],
                    "total_links": len(c["links"]),
                    "break_link": self.get_break_link(name),
                }
                for name, c in self.chains.items()
            },
        }
