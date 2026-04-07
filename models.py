# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the CiteGuardian Environment.

CiteGuardian simulates a peer-review / journal-editing workflow where an agent
audits a research paper for structural omissions, citation orphans, and
factual contradictions.
"""

from typing import Any, Literal, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

ActionType = Literal[
    "GO_TO",
    "SCAN_CITATIONS",
    "COMPARE_VALUES",
    "FLAG_ERROR",
    "SUBMIT",
]

ErrorType = Literal[
    "STRUCTURAL_ERROR",
    "ORPHAN_CITATION",
    "LOGICAL_INCONSISTENCY",
]


class CiteguardianAction(Action):
    """
    Unified action for the CiteGuardian environment.

    Depending on `action_type`, only the relevant fields are used:
      - GO_TO          → section_name
      - SCAN_CITATIONS → (no extra fields)
      - COMPARE_VALUES → val1, val2
      - FLAG_ERROR     → error_type, text_snippet
      - SUBMIT         → (no extra fields)
    """

    action_type: ActionType = Field(..., description="Which tool the agent is invoking")

    # GO_TO
    section_name: Optional[str] = Field(
        default=None, description="Target section name for GO_TO"
    )

    # COMPARE_VALUES
    val1: Optional[str] = Field(
        default=None, description="First value for COMPARE_VALUES"
    )
    val2: Optional[str] = Field(
        default=None, description="Second value for COMPARE_VALUES"
    )

    # FLAG_ERROR
    error_type: Optional[ErrorType] = Field(
        default=None, description="Category of error being flagged"
    )
    text_snippet: Optional[str] = Field(
        default=None, description="Relevant text excerpt supporting the flag"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class CiteguardianObservation(Observation):
    model_config = ConfigDict(extra="ignore")

    """
    Observation returned after each step.

    current_view   – up to ~1000 chars of the current section text.
    metadata       – word count, detected citation markers, available sections.
    audit_log      – list of (step, action_type, detail) tuples already taken.
    tool_result    – output of SCAN_CITATIONS or COMPARE_VALUES, if applicable.
    message        – human-readable feedback from the environment.
    task_level     – 'A', 'B', or 'C' (difficulty).
    """

    current_view: str = Field(default="", description="Text window of the current section")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="word_count, citation_markers, available_sections",
    )
    audit_log: list[dict[str, Any]] = Field(
        default_factory=list, description="History of actions taken this episode"
    )
    tool_result: Optional[Any] = Field(
        default=None, description="Output of SCAN_CITATIONS or COMPARE_VALUES"
    )
    message: str = Field(default="", description="Environment feedback message")
    task_level: str = Field(default="A", description="Task difficulty: A, B, or C")
