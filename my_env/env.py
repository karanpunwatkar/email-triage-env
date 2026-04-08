from my_env.models import EmailObservation, EmailAction
from my_env.tasks import get_task
from my_env.grader import grade_action

class EmailEnv:

    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.task = get_task(task_name)
        self.done = False
        self.current_step = 0
        self.history = []

    async def reset(self):
        self.done = False
        self.current_step = 0
        self.history = []
        obs = EmailObservation(**self.task["email"])
        return {"observation": obs, "reward": 0.0, "done": False, "info": {}}

    async def step(self, action: EmailAction):
        self.current_step += 1
        reward, done, info = grade_action(
            self.task, action, step=self.current_step, history=self.history
        )
        self.done = done

        obs = EmailObservation(
            **self.task["email"],
            last_action_result=f"reward={reward:.2f}"
        )

        self.history.append({"step": self.current_step, "action": action.dict(), "reward": reward})
        return {"observation": obs, "reward": reward, "done": done, "info": info}

    async def state(self):
        return {
            "task": self.task_name,
            "step": self.current_step,
            "done": self.done
        }