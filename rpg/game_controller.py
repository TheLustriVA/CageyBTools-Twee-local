from __future__ import annotations
from typing import List, Tuple
from .agent import Agent  # relative import

class GameController:
    def __init__(self, dm: Agent, players: List[Agent]):
        self.dm = dm
        self.players = players
        self.turn_history: List[Tuple[str, str]] = []

    def run_round(self):
        context = self._build_context()
        dm_narration = self.dm.take_turn(context)
        self.turn_history.append(("DM", dm_narration))
        for player in self.players:
            context = self._build_context()
            action = player.take_turn(context)
            self.turn_history.append((player.name, action))

    def _build_context(self) -> str:
        return "\n\n".join(f"{name}: {text}" for name, text in self.turn_history[-10:])
