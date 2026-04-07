# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Citeguardian Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CiteguardianAction, CiteguardianObservation


class CiteguardianEnv(
    EnvClient[CiteguardianAction, CiteguardianObservation, State]
):
    """
    Client for the CiteGuardian Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each instance gets its own dedicated session on the server.

    Example:
        >>> async with CiteguardianEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     result = await client.step(CiteguardianAction(action_type="SUBMIT"))

    Example with Docker:
        >>> client = await CiteguardianEnv.from_docker_image("openenv-citeguardian")
        >>> result = await client.reset()
        >>> result = await client.step(CiteguardianAction(action_type="GO_TO", section_name="Methods"))
    """

    def _step_payload(self, action: CiteguardianAction) -> Dict:
        """Convert CiteguardianAction to JSON payload for the step request."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CiteguardianObservation]:
        """Parse server response into StepResult[CiteguardianObservation]."""
        obs_data = payload.get("observation", payload)
        observation = CiteguardianObservation(
            current_view=obs_data.get("current_view", ""),
            metadata=obs_data.get("metadata", {}),
            audit_log=obs_data.get("audit_log", []),
            tool_result=obs_data.get("tool_result"),
            message=obs_data.get("message", ""),
            task_level=obs_data.get("task_level", "A"),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
