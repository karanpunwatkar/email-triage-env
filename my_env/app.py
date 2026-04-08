from fastapi import FastAPI
import asyncio
from my_env.env import EmailEnv
from my_env.models import EmailAction

app = FastAPI()

env = EmailEnv(task_name="easy")

@app.api_route("/reset", methods=["GET", "POST"])
async def reset():
    result = await env.reset()
    return {
        "observation": result["observation"].model_dump(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"]
    }

@app.post("/step")
async def step(action: dict):
    action_obj = EmailAction(**action)
    result = await env.step(action_obj)

    return {
        "observation": result["observation"].model_dump(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"]
    }

@app.get("/")
def home():
    return {"status": "Email Triage Env Running"}