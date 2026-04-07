# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CiteGuardian Environment Implementation.

Simulates a professional peer-review / journal-editing workflow.
The agent audits a research paper for:
  Task A (Easy)   – Structural omissions (missing mandatory section)
  Task B (Medium) – Citation orphans (cited but not in References, or vice-versa)
  Task C (Hard)   – Factual contradictions across sections (numeric mismatches)

Reward structure (cumulative, max 1.0):
  +0.02  Exploration  – first visit to each required section
  +0.30  Accuracy     – correct FLAG_ERROR on a seeded mistake
  -0.10  False Pos.   – FLAG_ERROR where no error exists
  -0.01  Efficiency   – every step taken
  +0.10–0.40 Completion – SUBMIT after finding all errors (scales with recall)
  Clamped to 1.0 on perfect run (100 % recall, 0 false positives).
"""

import random
import re
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CiteguardianAction, CiteguardianObservation
except ImportError:
    from models import CiteguardianAction, CiteguardianObservation


# ---------------------------------------------------------------------------
# Paper templates
# ---------------------------------------------------------------------------

_TASK_A_PAPER = {
    "Abstract": (
        "We present a novel deep-learning approach for protein folding prediction. "
        "Our model achieves state-of-the-art accuracy on the CASP14 benchmark. [1][3]"
    ),
    "Introduction": (
        "Protein structure prediction has long been a grand challenge in biology. "
        "Recent advances in transformer architectures [1] have enabled breakthroughs. "
        "This paper extends the work of Jumper et al. (2021) [2] by incorporating "
        "multi-scale attention. We recruited 120 domain experts to evaluate outputs."
    ),
    "Methods": (
        "We trained a 48-layer transformer on the UniRef90 database. "
        "Training used 128 TPU-v4 chips for 14 days. "
        "Hyperparameters were tuned via Bayesian optimisation [3]. "
        "We recruited 120 subjects for the human-evaluation study."
    ),
    # "Results" section intentionally MISSING — this is the seeded error
    "Discussion": (
        "Our approach outperforms prior methods on all benchmarks. "
        "Limitations include high compute cost and limited generalisation to "
        "intrinsically disordered proteins."
    ),
    "References": (
        "[1] Vaswani et al. (2017) Attention is All You Need. NeurIPS.\n"
        "[2] Jumper et al. (2021) Highly accurate protein structure prediction. Nature.\n"
        "[3] Snoek et al. (2012) Practical Bayesian Optimization. NeurIPS."
    ),
}

_TASK_A_ERRORS = [
    {
        "id": "err_A1",
        "error_type": "STRUCTURAL_ERROR",
        "section": None,  # not section-specific
        "hint": "Results",  # the missing section name
        "description": "The mandatory 'Results' section is absent from the paper.",
    }
]

# ----

_TASK_B_PAPER = {
    "Abstract": (
        "We introduce CiteNet, a citation-graph neural network. "
        "Experiments on three benchmarks confirm superior performance [1][2]."
    ),
    "Introduction": (
        "Citation networks encode rich relational information [1]. "
        "Prior work by Kipf & Welling (2017) [2] used GCNs; we extend this "
        "with dynamic edge weighting. See also the survey by Hamilton et al. [4]."
        # [4] is cited here but NOT in References — orphan citation
    ),
    "Methods": (
        "Our model stacks three graph-attention layers [3]. "
        "We use the Adam optimiser with lr=0.001. "
        "Datasets: Cora, Citeseer, PubMed."
    ),
    "Results": (
        "CiteNet achieves 89.2 % accuracy on Cora, surpassing all baselines. "
        "Full results are in Table 1."
    ),
    "Discussion": (
        "The dynamic edge weighting is the key contributor to performance gains. "
        "Future work will explore temporal citation graphs."
    ),
    "References": (
        "[1] Perozzi et al. (2014) DeepWalk. KDD.\n"
        "[2] Kipf & Welling (2017) Semi-supervised classification with GCNs. ICLR.\n"
        "[3] Velickovic et al. (2018) Graph Attention Networks. ICLR.\n"
        # [4] Hamilton et al. intentionally MISSING from References
        # [5] listed here but never cited in text — extra orphan
        "[5] Xu et al. (2019) How Powerful are Graph Neural Networks? ICLR."
    ),
}

_TASK_B_ERRORS = [
    {
        "id": "err_B1",
        "error_type": "ORPHAN_CITATION",
        "section": "Introduction",
        "hint": "[4]",
        "description": "[4] is cited in Introduction but has no entry in References.",
    },
    {
        "id": "err_B2",
        "error_type": "ORPHAN_CITATION",
        "section": "References",
        "hint": "[5]",
        "description": "[5] appears in References but is never cited in the paper body.",
    },
]

# ----

_TASK_C_PAPER = {
    "Abstract": (
        "We conduct a randomised controlled trial on a novel cognitive training "
        "programme. Results show significant improvement in working memory. [1][2]"
    ),
    "Introduction": (
        "Cognitive decline affects millions worldwide [1]. "
        "Intervention studies [2] show promise but lack rigorous controls. "
        "We address this gap with a pre-registered RCT."
    ),
    "Methods": (
        "We recruited 100 subjects aged 60–75 from three urban clinics. "
        "Participants were randomised 1:1 to treatment and control arms (50 each). "
        "Primary outcome: digit-span score at 12 weeks."
    ),
    "Results": (
        # Contradiction: Methods says 100 subjects, Results says 85
        "Table 1 shows data for 85 subjects who completed the 12-week assessment. "
        "The treatment group (n=44) showed a mean improvement of 2.3 points (p<0.001). "
        "No adverse events were recorded."
        # No explanation for the 15-subject drop — this is the seeded error
    ),
    "Discussion": (
        "The significant improvement supports the efficacy of the programme. "
        "Limitations include the urban-only sample and short follow-up period."
    ),
    "References": (
        "[1] WHO (2023) Global status report on the public health response to dementia.\n"
        "[2] Smith et al. (2021) Cognitive interventions in older adults. Lancet."
    ),
}

_TASK_C_ERRORS = [
    {
        "id": "err_C1",
        "error_type": "LOGICAL_INCONSISTENCY",
        "section": "Results",
        "hint": "85",
        "description": (
            "Methods states 100 subjects were recruited, but Results reports data "
            "for only 85 subjects with no explanation for the 15-subject discrepancy."
        ),
    }
]

_TASKS = [
    ("A", _TASK_A_PAPER, _TASK_A_ERRORS),
    ("B", _TASK_B_PAPER, _TASK_B_ERRORS),
    ("C", _TASK_C_PAPER, _TASK_C_ERRORS),
]

MANDATORY_SECTIONS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "References"]

# Reward constants
_EXPLORATION_REWARD = 0.02
_ACCURACY_REWARD = 0.30
_FALSE_POSITIVE_PENALTY = -0.10
_STEP_PENALTY = -0.01
_MAX_COMPLETION_BONUS = 0.40
_MIN_COMPLETION_BONUS = 0.10
_VIEW_MAX_CHARS = 1000


def _extract_citations(text: str) -> list[str]:
    """Return all citation markers like [1], [2], [12] found in text."""
    return list(dict.fromkeys(re.findall(r"\[\d+\]", text)))


class CiteguardianEnvironment(Environment):
    """
    CiteGuardian: a peer-review RL environment.

    The agent navigates a research paper, uses audit tools, flags errors,
    and submits when done. Rewards are cumulative and capped at 1.0.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # These are set properly in reset()
        self._task_level: str = "A"
        self._paper: dict[str, str] = {}
        self._hidden_errors: list[dict] = []
        self._current_section: str = ""
        self._visited_sections: set[str] = set()
        self._flagged_errors: list[dict] = []  # what the agent has flagged
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> CiteguardianObservation:
        level, paper, errors = random.choice(_TASKS)
        self._task_level = level
        self._paper = dict(paper)
        self._hidden_errors = list(errors)
        self._current_section = list(self._paper.keys())[0]
        self._visited_sections = set()
        self._flagged_errors = []
        self._cumulative_reward = 0.0
        self._done = False
        self._audit_log = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(
            message=f"[Task {level}] Paper loaded. Begin your audit. "
                    f"Available sections: {list(self._paper.keys())}",
            tool_result=None,
        )

    def step(self, action: CiteguardianAction) -> CiteguardianObservation:  # type: ignore[override]
        if self._done:
            return self._make_observation(
                message="Episode already finished. Call reset() to start a new one.",
                tool_result=None,
            )

        self._state.step_count += 1
        # Step penalty every action
        self._cumulative_reward += _STEP_PENALTY

        atype = action.action_type
        tool_result = None
        message = ""

        if atype == "GO_TO":
            message, tool_result = self._handle_go_to(action)

        elif atype == "SCAN_CITATIONS":
            message, tool_result = self._handle_scan_citations()

        elif atype == "COMPARE_VALUES":
            message, tool_result = self._handle_compare_values(action)

        elif atype == "FLAG_ERROR":
            message, tool_result = self._handle_flag_error(action)

        elif atype == "SUBMIT":
            message, tool_result = self._handle_submit()

        else:
            message = f"Unknown action_type '{atype}'."

        self._audit_log.append(
            {"step": self._state.step_count, "action": atype, "detail": message}
        )

        obs = self._make_observation(message=message, tool_result=tool_result)
        obs.reward = round(self._cumulative_reward, 4)
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_go_to(self, action: CiteguardianAction):
        section = action.section_name or ""
        if section not in self._paper:
            return (
                f"Section '{section}' not found. "
                f"Available: {list(self._paper.keys())}",
                None,
            )
        self._current_section = section
        # Exploration reward — first visit only
        if section not in self._visited_sections:
            self._visited_sections.add(section)
            self._cumulative_reward += _EXPLORATION_REWARD
            extra = f" [+{_EXPLORATION_REWARD} exploration reward]"
        else:
            extra = ""
        return f"Navigated to '{section}'.{extra}", None

    def _handle_scan_citations(self):
        text = self._paper.get(self._current_section, "")
        citations = _extract_citations(text)
        return (
            f"SCAN_CITATIONS in '{self._current_section}': found {len(citations)} marker(s).",
            citations,
        )

    def _handle_compare_values(self, action: CiteguardianAction):
        v1 = str(action.val1 or "").strip()
        v2 = str(action.val2 or "").strip()
        try:
            n1, n2 = float(v1), float(v2)
            conflict = abs(n1 - n2) > 1e-9
        except ValueError:
            conflict = v1.lower() != v2.lower()

        result = {
            "val1": v1,
            "val2": v2,
            "conflict_detected": conflict,
        }
        msg = (
            f"COMPARE_VALUES: '{v1}' vs '{v2}' → "
            f"{'CONFLICT DETECTED' if conflict else 'no conflict'}."
        )
        return msg, result

    def _handle_flag_error(self, action: CiteguardianAction):
        etype = action.error_type or ""
        snippet = action.text_snippet or ""

        # Check against hidden errors
        matched = self._match_hidden_error(etype, snippet)

        if matched:
            if matched["id"] in [f["matched_id"] for f in self._flagged_errors if "matched_id" in f]:
                # Already flagged this one — treat as redundant false positive
                self._cumulative_reward += _FALSE_POSITIVE_PENALTY
                return (
                    f"FLAG_ERROR: '{etype}' — already flagged. "
                    f"Duplicate flag penalised ({_FALSE_POSITIVE_PENALTY}).",
                    None,
                )
            self._cumulative_reward += _ACCURACY_REWARD
            self._flagged_errors.append(
                {"error_type": etype, "snippet": snippet, "matched_id": matched["id"]}
            )
            return (
                f"FLAG_ERROR: '{etype}' — CORRECT. "
                f"Matched seeded error '{matched['id']}'. "
                f"[+{_ACCURACY_REWARD} accuracy reward]",
                None,
            )
        else:
            self._cumulative_reward += _FALSE_POSITIVE_PENALTY
            self._flagged_errors.append(
                {"error_type": etype, "snippet": snippet, "matched_id": None}
            )
            return (
                f"FLAG_ERROR: '{etype}' — FALSE POSITIVE. "
                f"No matching seeded error found. [{_FALSE_POSITIVE_PENALTY} penalty]",
                None,
            )

    def _handle_submit(self):
        self._done = True
        total_errors = len(self._hidden_errors)
        correct_flags = len([f for f in self._flagged_errors if f.get("matched_id")])
        false_positives = len([f for f in self._flagged_errors if not f.get("matched_id")])

        recall = correct_flags / total_errors if total_errors > 0 else 0.0

        # Completion bonus scales linearly with recall
        bonus = _MIN_COMPLETION_BONUS + recall * (_MAX_COMPLETION_BONUS - _MIN_COMPLETION_BONUS)
        self._cumulative_reward += bonus

        # Perfect run clamp
        if recall == 1.0 and false_positives == 0:
            self._cumulative_reward = 1.0
            verdict = "PERFECT AUDIT"
        else:
            self._cumulative_reward = min(self._cumulative_reward, 1.0)
            verdict = "AUDIT COMPLETE"

        msg = (
            f"SUBMIT — {verdict}. "
            f"Errors found: {correct_flags}/{total_errors}. "
            f"False positives: {false_positives}. "
            f"Final reward: {round(self._cumulative_reward, 4)}."
        )
        return msg, {
            "recall": recall,
            "false_positives": false_positives,
            "completion_bonus": round(bonus, 4),
            "final_reward": round(self._cumulative_reward, 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_hidden_error(self, etype: str, snippet: str) -> dict | None:
        """
        Try to match a FLAG_ERROR call against the hidden error list.
        Matching rules:
          - error_type must match exactly.
          - snippet must contain the error's hint string (case-insensitive).
        """
        for err in self._hidden_errors:
            if err["error_type"] != etype:
                continue
            hint = err.get("hint", "")
            if hint and hint.lower() not in snippet.lower():
                continue
            return err
        return None

    def _make_observation(self, message: str, tool_result) -> CiteguardianObservation:
        section_text = self._paper.get(self._current_section, "")
        view = section_text[:_VIEW_MAX_CHARS]

        # Build full-paper citation index for metadata
        all_citations: list[str] = []
        for text in self._paper.values():
            all_citations.extend(_extract_citations(text))
        all_citations = list(dict.fromkeys(all_citations))

        metadata = {
            "current_section": self._current_section,
            "available_sections": list(self._paper.keys()),
            "word_count": len(section_text.split()),
            "citation_markers_in_view": _extract_citations(section_text),
            "all_paper_citations": all_citations,
            "visited_sections": list(self._visited_sections),
            "flags_raised": len(self._flagged_errors),
            "step": self._state.step_count,
        }

        return CiteguardianObservation(
            current_view=view,
            metadata=metadata,
            audit_log=list(self._audit_log),
            tool_result=tool_result,
            message=message,
            task_level=self._task_level,
            done=self._done,
            reward=round(self._cumulative_reward, 4),
        )
