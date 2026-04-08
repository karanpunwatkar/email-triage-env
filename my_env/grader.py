from my_env.models import EmailAction
from typing import Tuple, Dict, Any

def grade_action(task: dict, action: EmailAction, step: int = 1, history: list = []) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Return: reward (0.0-1.0), done (True/False), info dict
    Stepwise grading for OpenEnv environment.
    """
    reward = 0.0
    done = False
    info: Dict[str, Any] = {}

    # Step 1: classify priority
    if step == 1:
        if action.priority == task.get("expected_priority", "low"):
            reward = 0.8
        else:
            reward = 0.0
        done = False

    # Step 2: choose main action
    elif step == 2:
        if action.action_type == task.get("expected_action", "acknowledge"):
            reward = 0.8
        else:
            reward = 0.0
        done = False

    # Step 3: compose reply (only for hard task)
    elif step == 3:
        expected_reply = task.get("expected_reply", "")
        if action.content == expected_reply:
            reward = 1.0
        else:
            # Partial reward if reply contains any words from expected
            correct_words = sum(word in action.content for word in expected_reply.split())
            total_words = len(expected_reply.split()) or 1
            reward = round((correct_words / total_words), 2)
        done = True

    info["step"] = step
    info["expected"] = task.get("expected_reply", "")
    return reward, done, info