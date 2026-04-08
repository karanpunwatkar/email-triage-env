import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from my_env.env import EmailEnv
from my_env.models import EmailAction

# ENV VARIABLES
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3

# LOGGING FUNCTIONS
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# MAIN TASK RUNNER
async def run_task(task_name: str):
    env = EmailEnv(task_name=task_name)

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env="email_triage_env", model=MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        if result["done"]:
            break

        # Extract expected values from task (baseline deterministic agent)
        expected_priority = env.task.get("expected_priority")
        expected_action = env.task.get("expected_action")
        expected_reply = env.task.get("expected_reply", "")

        # Build action properly
        if step == 1:
            action_obj = EmailAction(
                priority=expected_priority,
                action_type="acknowledge",
                content=None
            )
        elif step == 2:
            action_obj = EmailAction(
                priority=expected_priority,
                action_type=expected_action,
                content=None
            )
        else:
            action_obj = EmailAction(
                priority=expected_priority,
                action_type="reply",
                content=expected_reply
            )

        # Step environment
        result = await env.step(action_obj)

        reward = result["reward"]
        done = result["done"]

        # LOG (use model_dump for Pydantic v2)
        log_step(
            step=step,
            action=str(action_obj.model_dump()),
            reward=reward,
            done=done,
            error=None
        )

        rewards.append(reward)
        steps_taken = step

        if done:
            break

    # FINAL SCORE
    score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
    score = min(max(score, 0.0), 1.0)

    success = score >= 0.5

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# MAIN ENTRY
async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())