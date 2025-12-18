from typing import List

def setup_training(self):
    """
    Dummy setup_training for rule_based_agent.
    This function does nothing but satisfies the requirement of main.py logic.
    """
    pass

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Dummy game_events_occurred.
    The rule_based_agent does not learn, so it ignores all events.
    """
    pass

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Dummy end_of_round.
    No model update happens here.
    """
    pass
