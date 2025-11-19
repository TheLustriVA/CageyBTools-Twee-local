from __future__ import annotations
from typing import Optional
from .llama_server_session import LlamaServerSession  # relative import

class Agent:
    def __init__(self, name: str, character_sheet: str, 
                 server_url: str, model_type: str):
        self.name = name
        self.character_sheet = character_sheet
        self.model_type = model_type  # "fiction" or "reasoning"
        self.server_url = server_url  # store it!
        self.session: Optional[LlamaServerSession] = None

    def take_turn(self, game_state: str) -> str:
        """Generate this agent's action/response."""
        if self.session is None:
            # keep a persistent session per Agent
            self.session = LlamaServerSession(self.server_url)
            self.session.connect()
            self.session.system_prompt(f"You are {self.name}. {self.character_sheet}")
        return self.session.prompt(game_state)
