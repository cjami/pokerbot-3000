"""Internal poker agent skeletons owned by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot_3000.domain.models import ActionType, ClientId, PokerAction, PrivateAgentState, PublicGameState


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Static configuration for an orchestrator-owned agent."""

    agent_id: str
    seat: int
    client_id: ClientId
    display_name: str
    personality: str


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """Internal agent decision plus presentation instructions."""

    action: PokerAction
    speech: str
    reaction: str
    emotion: str
    gesture: str


class StubPokerAgent:
    """Deterministic placeholder until Gemma-backed decisions are wired in."""

    def decide(
        self,
        profile: AgentProfile,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentTurn:
        """Pick a legal placeholder action from public and private state."""
        if ActionType.CALL in public_state.legal_actions:
            action = PokerAction(type=ActionType.CALL, amount=public_state.current_bet_to_call)
            phrase = f"{profile.display_name} calls with {len(private_state.hole_cards)} private cards known."
        elif ActionType.CHECK in public_state.legal_actions:
            action = PokerAction(type=ActionType.CHECK)
            phrase = f"{profile.display_name} checks."
        else:
            action = PokerAction(type=ActionType.FOLD)
            phrase = f"{profile.display_name} folds."

        emotion, gesture = _presentation_for_action(action.type)
        return AgentTurn(action=action, speech=phrase, reaction="announce_action", emotion=emotion, gesture=gesture)


def _presentation_for_action(action_type: ActionType) -> tuple[str, str]:
    presentations = {
        ActionType.CHECK: ("calm", "nod"),
        ActionType.CALL: ("calm", "nod"),
        ActionType.BET: ("confident", "lean_in"),
        ActionType.RAISE_TO: ("confident", "lean_in"),
        ActionType.ALL_IN: ("celebrate", "big_nod"),
        ActionType.FOLD: ("sad", "look_down"),
    }
    return presentations.get(action_type, ("confused", "tilt"))
