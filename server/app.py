from fastapi import FastAPI
import uvicorn
from my_env.env import EmailEnv
from my_env.models import EmailAction

app = FastAPI()
env = EmailEnv(task_name="easy")


@app.api_route("/reset", methods=["GET", "POST"])
async def reset():
    result = await env.reset()
    return {
        "observation": result["observation"].model_dump(),
        "reward": float(result["reward"]),
        "done": bool(result["done"]),
        "info": result["info"] or {}
    }


@app.post("/step")
async def step(action: EmailAction):
    result = await env.step(action)
    return {
        "observation": result["observation"].model_dump(),
        "reward": float(result["reward"]),
        "done": bool(result["done"]),
        "info": result["info"] or {}
    }


@app.get("/state")
async def state():
    return await env.state()


@app.get("/")
def home():
    return {"status": "Email Triage Env Running"}


# ✅ REQUIRED FOR VALIDATOR
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED ENTRYPOINT
if __name__ == "__main__":
    main()