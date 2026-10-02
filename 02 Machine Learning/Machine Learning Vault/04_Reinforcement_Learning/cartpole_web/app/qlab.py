"""Q-Table Lab: เดิน Q-Learning ทีละ step แล้วส่งรายละเอียดทุกขั้นให้หน้าเว็บ

ทุก step ส่ง: state → bin index, ε-greedy (สุ่มหรือเลือกค่าสูงสุด), ค่าที่แทนในสมการ
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
และ "slice" ของ Q-table (θ × θ̇) ที่ state นั้นอยู่ เพื่อวาด heatmap
"""
import asyncio
from pathlib import Path

import gymnasium as gym
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .agents import QLearningAgent

QTABLE = Path(__file__).parent / "models" / "qtable.npy"

router = APIRouter()


def r3(v):
    return round(float(v), 3)


class Lab:
    def __init__(self):
        self.fps = 5
        self.running = False
        self.busy = False
        self.reset("scratch")

    def reset(self, source):
        if source == "pretrained" and QTABLE.exists():
            self.agent = QLearningAgent.load(QTABLE)
        else:
            source, self.agent = "scratch", QLearningAgent()
        self.source = source
        self.visits = np.zeros(self.agent.q.shape, dtype=int)  # จำนวนครั้งที่แต่ละช่องถูก update
        self.env = gym.make("CartPole-v1")
        self.obs, _ = self.env.reset()
        self.steps, self.ep_reward, self.episode, self.total_updates = 0, 0.0, 0, 0
        self.history = []

    def slice(self, xb, xdb):
        """Q-table ตัดตาม (x bin, ẋ bin) → ตาราง θ (8) × θ̇ (12) × action (2)"""
        return {
            "x": int(xb), "xd": int(xdb),
            "q": np.round(self.agent.q[xb, xdb], 2).tolist(),
            "visits": self.visits[xb, xdb].sum(axis=-1).tolist(),
        }

    def info(self):
        a = self.agent
        return {"type": "info", "source": self.source, "epsilon": r3(a.epsilon), "alpha": r3(a.alpha),
                "gamma": a.gamma, "episode": self.episode, "total_updates": self.total_updates,
                "nonzero": int(np.count_nonzero(a.q)), "cells": int(a.q.size), "history": self.history[-200:],
                "bins": QLearningAgent.BINS.tolist(), "lows": QLearningAgent.LOWS.tolist(),
                "highs": QLearningAgent.HIGHS.tolist()}

    def step(self):
        ag = self.agent
        obs = self.obs
        s = ag.discretize(obs)
        eps, alpha, gamma = ag.epsilon, ag.alpha, ag.gamma

        # 1) ε-greedy: สุ่มเลข u ถ้า u < ε → explore (สุ่ม action) ไม่งั้น exploit (argmax Q)
        u = float(np.random.rand())
        explore = u < eps
        q_s = ag.q[s].copy()
        action = int(self.env.action_space.sample()) if explore else int(np.argmax(q_s))

        # 2) ทำ action → ได้ reward และ state ถัดไป
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        s2 = ag.discretize(next_obs)
        q_s2 = ag.q[s2].copy()

        # 3) Bellman update (ถ้า terminated ไม่มีอนาคต → max Q(s',·) = 0)
        max_q2 = 0.0 if terminated else float(np.max(q_s2))
        old = float(ag.q[s + (action,)])
        target = reward + gamma * max_q2
        td_error = target - old
        new = old + alpha * td_error
        ag.q[s + (action,)] = new
        self.visits[s + (action,)] += 1
        self.total_updates += 1

        self.steps += 1
        self.ep_reward += reward
        detail = {
            "type": "step", "episode": self.episode, "step": self.steps, "ep_reward": self.ep_reward,
            "obs": [r3(v) for v in obs], "next_obs": [r3(v) for v in next_obs],
            "s": [int(i) for i in s], "s2": [int(i) for i in s2],
            "epsilon": r3(eps), "u": r3(u), "explore": explore, "action": action,
            "q_s": [r3(v) for v in q_s], "q_s2": [r3(v) for v in q_s2],
            "reward": float(reward), "alpha": r3(alpha), "gamma": gamma,
            "max_q2": r3(max_q2), "old": r3(old), "target": r3(target), "td_error": r3(td_error), "new": r3(new),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "slice": self.slice(s[0], s[1]),
        }

        if terminated or truncated:
            ag.episodes += 1
            self.history.append(self.ep_reward)
            self.episode += 1
            self.obs, _ = self.env.reset()
            self.steps, self.ep_reward = 0, 0.0
        else:
            self.obs = next_obs
        return detail


async def play_loop(ws: WebSocket, lab: Lab):
    while lab.running:
        if not lab.busy:
            d = lab.step()
            await ws.send_json(d)
            if d["terminated"] or d["truncated"]:
                await ws.send_json(lab.info())
        await asyncio.sleep(1 / lab.fps)


@router.websocket("/ws/qlab")
async def qlab_ws(ws: WebSocket):
    await ws.accept()
    lab = Lab()
    task = None
    await ws.send_json(lab.info())
    await ws.send_json({"type": "slice", "slice": lab.slice(*lab.agent.discretize(lab.obs)[:2])})
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
                lab.fps = max(1, min(200, int(msg.get("fps", 5))))
            elif cmd == "reset":
                lab.running = False
                lab.reset(msg.get("source", "scratch"))
                await ws.send_json(lab.info())
                await ws.send_json({"type": "slice", "slice": lab.slice(*lab.agent.discretize(lab.obs)[:2])})
            elif cmd == "slice":
                await ws.send_json({"type": "slice", "slice": lab.slice(int(msg["x"]) % 3, int(msg["xd"]) % 3)})
            elif cmd == "fast" and not lab.busy:
                # ฝึกเร็วแบบ headless (ไม่ส่งรายละเอียดทีละ step)
                lab.busy = True
                n = max(1, min(3000, int(msg.get("episodes", 100))))
                before = lab.agent.q.copy()
                rewards = await asyncio.to_thread(lab.agent.train, gym.make("CartPole-v1"), n)
                lab.visits += (lab.agent.q != before).astype(int)  # นับอย่างน้อย 1 ครั้งให้ช่องที่เปลี่ยน
                lab.history += rewards
                lab.episode += n
                lab.busy = False
                await ws.send_json(lab.info())
                await ws.send_json({"type": "slice", "slice": lab.slice(*lab.agent.discretize(lab.obs)[:2])})
    except WebSocketDisconnect:
        pass
    finally:
        lab.running = False
        if task:
            task.cancel()
