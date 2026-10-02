"""WebSocket /ws/maze: เดิน DQN ใน maze ทีละ step แล้วส่งรายละเอียดให้หน้าเว็บ"""
import asyncio

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .maze_dqn import MAZES, DQNAgent, Maze, run_episode

router = APIRouter()


def rnd(a, d=3):
    return np.round(np.asarray(a, dtype=float), d).tolist()


class MazeLab:
    def __init__(self):
        self.fps, self.running, self.busy, self.mode = 8, False, False, "train"
        self.reset("note")

    def reset(self, name):
        self.name = name if name in MAZES else "note"
        self.maze = Maze(MAZES[self.name])
        self.agent = DQNAgent(self.maze.n_states)
        self.history = []  # [reward, steps, reached]
        self.pos, self.ep_reward = self.maze.reset(), 0.0

    def rc(self, idx):
        return [int(idx) // self.maze.w, int(idx) % self.maze.w]

    def policy(self):
        """Q(s,·) ของทุกช่องจาก online network (ใช้วาดลูกศร/สีบนแผนที่)"""
        return rnd(self.agent.online.forward(np.eye(self.maze.n_states, dtype=np.float32)), 3)

    def info(self):
        m, ag = self.maze, self.agent
        return {"type": "info", "name": self.name, "grid": ["".join(r) for r in m.grid], "h": m.h, "w": m.w,
                "start": m.start, "goal": m.goal, "shortest": m.shortest_path_len(), "max_steps": m.max_steps,
                "n_states": m.n_states, "table_size": m.n_states * 4, "n_params": ag.online.n_params,
                "hidden": ag.online.W[0].shape[1], "gamma": ag.gamma, "batch": ag.batch, "buffer_max": ag.memory.maxlen,
                "target_every": ag.target_every, "lr": ag.online.lr,
                "epsilon": round(ag.epsilon, 3), "episodes": ag.episodes, "buffer": len(ag.memory),
                "learn_steps": ag.learn_steps, "history": self.history[-300:], "mode": self.mode,
                "pos": self.pos, "policy": self.policy()}

    def step(self):
        m, ag = self.maze, self.agent
        learn = self.mode == "train"
        x = m.encode(self.pos)
        acts = ag.online.forward(x[None], keep=True)
        q = acts[-1][0]
        a, explore, u = ag.act(x, explore=learn)
        before = self.pos
        pos2, r, term, trunc, bumped = m.step(a)
        x2 = m.encode(pos2)
        replay = None
        if learn:
            ag.remember(x, a, r, x2, term)
            replay = ag.replay()
            if replay:
                s = replay["sample"]
                s["s"], s["s2"] = self.rc(s["s"]), self.rc(s["s2"])
                s["q_next"] = rnd(s["q_next"])
        self.ep_reward += r
        d = {"type": "step", "mode": self.mode, "episode": ag.episodes, "step": m.steps,
             "pos": list(before), "pos2": list(pos2), "action": a, "explore": explore, "u": round(u, 3),
             "epsilon": round(ag.epsilon, 3) if learn else 0, "q": rnd(q), "h1": rnd(acts[1][0], 2),
             "h2": rnd(acts[2][0], 2), "reward": r, "bumped": bumped, "terminated": term, "truncated": trunc,
             "ep_reward": round(self.ep_reward, 3), "buffer": len(ag.memory), "learn_steps": ag.learn_steps,
             "target_in": ag.target_every - ag.learn_steps % ag.target_every, "replay": replay,
             "exp": [list(before), a, r, list(pos2), term], "policy": self.policy()}
        if term or trunc:
            self.history.append([round(self.ep_reward, 3), m.steps, bool(term)])
            if learn:
                ag.end_episode()
            self.pos, self.ep_reward = m.reset(), 0.0
        else:
            self.pos = pos2
        return d

    def fast(self, n):
        for _ in range(n):
            total, steps, reached = run_episode(self.maze, self.agent, learn=True)
            self.history.append([round(total, 3), steps, reached])
        self.pos, self.ep_reward = self.maze.reset(), 0.0


async def play_loop(ws, lab):
    while lab.running:
        if not lab.busy:
            d = lab.step()
            await ws.send_json(d)
            if d["terminated"] or d["truncated"]:
                await ws.send_json(lab.info())
        await asyncio.sleep(1 / lab.fps)


@router.websocket("/ws/maze")
async def maze_ws(ws: WebSocket):
    await ws.accept()
    lab, task = MazeLab(), None
    await ws.send_json(lab.info())
    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "step" and not lab.running and not lab.busy:
                d = lab.step()
                await ws.send_json(d)
                if d["terminated"] or d["truncated"]:
                    await ws.send_json(lab.info())
            elif cmd == "play" and not lab.running:
                lab.running = True
                task = asyncio.create_task(play_loop(ws, lab))
            elif cmd == "pause":
                lab.running = False
            elif cmd == "fps":
                lab.fps = max(1, min(120, int(msg.get("fps", 8))))
            elif cmd == "mode":
                lab.mode = "test" if msg.get("mode") == "test" else "train"
                lab.pos, lab.ep_reward = lab.maze.reset(), 0.0
                await ws.send_json(lab.info())
            elif cmd == "reset":
                lab.running = False
                lab.reset(msg.get("maze", lab.name))
                await ws.send_json(lab.info())
            elif cmd == "toggle" and not lab.running:
                # แก้กำแพง: network เดิมยังอยู่ → ดูว่า agent ปรับตัวกับ environment ใหม่ได้อย่างไร
                lab.maze.toggle_wall(int(msg["r"]), int(msg["c"]))
                if lab.maze.is_wall(*lab.pos):
                    lab.pos = lab.maze.reset()
                await ws.send_json(lab.info())
            elif cmd == "fast" and not lab.busy:
                lab.busy = True
                await asyncio.to_thread(lab.fast, max(1, min(2000, int(msg.get("episodes", 50)))))
                lab.busy = False
                await ws.send_json(lab.info())
    except WebSocketDisconnect:
        pass
    finally:
        lab.running = False
        if task:
            task.cancel()
