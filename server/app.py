from fastapi import FastAPI
from my_env.env import EmailEnv
from my_env.models import EmailAction

app = FastAPI()

# Global env instance
env = EmailEnv(task_name="easy")


# ✅ RESET (GET + POST allowed)
@app.api_route("/reset", methods=["GET", "POST"])
async def reset():
    result = await env.reset()

    return {
        "observation": result["observation"].model_dump(),
        "reward": float(result["reward"]),
        "done": bool(result["done"]),
        "info": result["info"] or {}
    }


# ✅ STEP (STRICT TYPE)
@app.post("/step")
async def step(action: EmailAction):   # 🔥 FIXED
    result = await env.step(action)

    return {
        "observation": result["observation"].model_dump(),
        "reward": float(result["reward"]),
        "done": bool(result["done"]),
        "info": result["info"] or {}
    }


# ✅ STATE (REQUIRED)
@app.get("/state")
async def state():
    return await env.state()


# ✅ HEALTH CHECK
@app.get("/")
def home():
    return {"status": "Email Triage Env Running"}