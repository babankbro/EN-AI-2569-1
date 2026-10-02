"""CartPole web server.

Python (Gymnasium) รัน simulation ฝั่ง server แล้วส่ง state ผ่าน WebSocket
ให้ browser วาดภาพบน <canvas> แบบ real-time
"""
import asyncio
import json
from pathlib import Path

import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agents import HeuristicAgent, QLearningAgent, RandomAgent
from .atari_ws import router as atari_router
from .maze_ws import router as maze_router
from .qlab import router as qlab_router

STATIC = Path(__file__).parent / "static"
MODELS = Path(__file__).parent / "models"

app = FastAPI(title="CartPole Web")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(qlab_router)
app.include_router(maze_router)
app.include_router(atari_router)


@app.get("/")
def index():
    return FileResponse(STATIC / "hub.html")


@app.get("/cartpole")
def cartpole_page():
    return FileResponse(STATIC / "cartpole.html")


@app.get("/maze")
def maze_page():
    return FileResponse(STATIC / "maze.html")


@app.get("/atari")
def atari_page():
    return FileResponse(STATIC / "atari.html")


@app.get("/qtable")
def qtable_page():
    return FileResponse(STATIC / "qtable.html")


@app.get("/api/history")
def history():
    """ผลการฝึก (reward ทุก episode + evaluation) ของโมเดลที่ฝึกไว้ล่วงหน้า"""
    path = MODELS / "history.json"
    return json.loads(path.read_text()) if path.exists() else {"rewards": [], "evals": []}


class Session:
    """สถานะของผู้ใช้ 1 connection (agent ของแต่ละคนแยกกัน)"""

    def __init__(self):
        self.agents = {"random": RandomAgent(), "heuristic": HeuristicAgent(), "qlearning": QLearningAgent()}
        if (MODELS / "qtable.npy").exists():
            # Q-table ที่ฝึกแล้ว (python -m app.train) → greedy เท่านั้น ไม่เรียนรู้ต่อ
            self.agents["pretrained"] = QLearningAgent.load(MODELS / "qtable.npy")
        self.policy = "pretrained" if "pretrained" in self.agents else "heuristic"
        self.learn = True
        self.fps = 50
        self.running = False
        self.busy = False  # กำลัง fast-train อยู่
        self.episode = 0

    @property
    def agent(self):
        return self.agents[self.policy]

    def info(self):
        q = self.agents["qlearning"]
        return {"type": "info", "policy": self.policy, "has_pretrained": "pretrained" in self.agents, "q_episodes": q.episodes, "epsilon": round(q.epsilon, 3)}


async def play_loop(ws: WebSocket, s: Session):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    step, total = 0, 0.0
    try:
        while s.running:
            if s.busy:
                await asyncio.sleep(0.05)
                continue
            learning = s.policy == "qlearning" and s.learn
            action = s.agent.act(obs, env, explore=learning)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if learning:
                s.agent.learn(obs, action, reward, next_obs, terminated)
            obs, step, total = next_obs, step + 1, total + reward

            await ws.send_json({
                "type": "state",
                "obs": [float(v) for v in obs],
                "action": action,
                "step": step,
                "reward": total,
                "episode": s.episode,
            })

            if terminated or truncated:
                if learning:
                    s.agent.episodes += 1
                await ws.send_json({"type": "episode_end", "episode": s.episode, "reward": total,
                                    "policy": s.policy, "truncated": truncated})
                await ws.send_json(s.info())
                s.episode += 1
                obs, _ = env.reset()
                step, total = 0, 0.0
                await asyncio.sleep(0.4)  # เว้นจังหวะให้เห็นว่าจบ episode
            await asyncio.sleep(1 / s.fps)
    finally:
        env.close()


async def fast_train(ws: WebSocket, s: Session, n: int):
    """ฝึก Q-Learning แบบไม่วาดภาพ เป็นช่วง ๆ ละ 50 episode แล้วส่งผลกลับไปวาดกราฟ"""
    s.busy = True
    env = gym.make("CartPole-v1")
    agent = s.agents["qlearning"]
    try:
        done = 0
        while done < n:
            chunk = min(50, n - done)
            rewards = await asyncio.to_thread(agent.train, env, chunk)
            done += chunk
            await ws.send_json({"type": "train_progress", "done": done, "total": n, "rewards": rewards})
            await ws.send_json(s.info())
    finally:
        env.close()
        s.busy = False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    s = Session()
    task = None
    await ws.send_json(s.info())
    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "start" and not s.running:
                s.running = True
                task = asyncio.create_task(play_loop(ws, s))
            elif cmd == "stop":
                s.running = False
            elif cmd == "config":
                if msg.get("policy") in s.agents:
                    s.policy = msg["policy"]
                s.learn = bool(msg.get("learn", s.learn))
                s.fps = max(1, min(500, int(msg.get("fps", s.fps))))
                await ws.send_json(s.info())
            elif cmd == "reset_q":
                s.agents["qlearning"] = QLearningAgent()
                await ws.send_json(s.info())
            elif cmd == "train" and not s.busy:
                n = max(1, min(5000, int(msg.get("episodes", 500))))
                asyncio.create_task(fast_train(ws, s, n))
    except WebSocketDisconnect:
        pass
    finally:
        s.running = False
        if task:
            task.cancel()
