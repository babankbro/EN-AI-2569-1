"""WebSocket /ws/atari: Breakout + DQN (ตาม Keras) ส่งภาพเกม, input 4 เฟรม, feature maps และการแทนค่า loss

ใช้ agent ตัวเดียวร่วมกันทั้ง server (replay memory ใช้ RAM มาก) ทุกหน้าที่เปิดอยู่จะเห็นสถานะเดียวกัน
"""
import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

MODELS = Path(__file__).parent / "models" / "atari"
CKPT = MODELS / "breakout_dqn.pt"
META = MODELS / "breakout_dqn.json"
# Keras ใช้ 100,000 แต่ 1 transition = 2 × (4×84×84) ≈ 56 KB → 100k ≈ 5.6 GB จึงลดลงสำหรับหน้าเว็บ
WEB_MEMORY = int(os.environ.get("ATARI_MEMORY", 10000))


def save_meta(agent):
    META.write_text(json.dumps({"frame_count": agent.frame_count, "episode_count": agent.episode_count,
                                "running_reward": agent.running_reward, "epsilon": agent.epsilon,
                                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}))


def read_meta():
    try:
        return json.loads(META.read_text()) if CKPT.exists() and META.exists() else None
    except (OSError, ValueError):
        return None


class AtariLab:
    def __init__(self):
        from .atari_dqn import AtariDQN, make_env  # import ช้า (torch) → โหลดเมื่อมีคนเปิดหน้านี้เท่านั้น
        self.env = make_env()
        self.agent = AtariDQN(memory_size=WEB_MEMORY)
        self.mode, self.fps, self.running, self.fast_running = "train", 30, False, False
        self.clients, self.lock, self.task = set(), threading.Lock(), None
        self.history = []  # {mode, reward, frame}
        self.message = ""
        self.new_episode()

    def new_episode(self):
        self.state, _ = self.env.reset()
        self.ep_reward, self.ep_steps = 0.0, 0
        self.lives = self.env.unwrapped.ale.lives()

    def reset_agent(self):
        from .atari_dqn import AtariDQN
        self.agent = AtariDQN(memory_size=WEB_MEMORY)
        self.history = []
        self.new_episode()

    # ---------- 1 step ----------
    def step(self, render=True):
        from .atari_dqn import ACTION_NAMES, feature_grid, heuristic_action, jpg_b64, png_b64
        ag, st = self.agent, self.state
        rgb = self.env.unwrapped.ale.getScreenRGB()
        eps_before = ag.epsilon
        if self.mode == "train":
            action, reason = ag.act(st, train=True)
        elif self.mode == "greedy":
            action, reason = ag.act(st, train=False)
        elif self.mode == "heuristic":
            action, reason = heuristic_action(rgb), "heuristic"
        else:
            action, reason = random.randrange(4), "random"

        s2, reward, terminated, truncated, _ = self.env.step(action)
        self.ep_steps += 1
        details = synced = None
        if self.mode == "train":
            ag.remember(st, action, reward, s2, terminated)
            details, synced = ag.learn()
        self.ep_reward += reward
        lives = self.env.unwrapped.ale.lives()
        lost_life, self.lives = lives < self.lives, lives
        ended = terminated or truncated or self.ep_steps >= ag.cfg["max_steps_per_episode"]

        payload = None
        if render:
            q, acts = ag.q_values(st, keep=True)
            c1, top1 = feature_grid(acts["conv1"][0], 8)
            c2, _ = feature_grid(acts["conv2"][0], 8)
            c3, _ = feature_grid(acts["conv3"][0], 8)
            dense = acts["dense"][0].numpy()
            payload = {
                "type": "step", "mode": self.mode, "action": action, "action_name": ACTION_NAMES[action],
                "reason": reason, "epsilon": round(eps_before, 5), "reward": float(reward),
                "ep_reward": self.ep_reward, "ep_steps": self.ep_steps, "lives": lives, "lost_life": lost_life,
                "terminated": bool(terminated), "ended": bool(ended),
                "frame_count": ag.frame_count, "episode": ag.episode_count, "memory": len(ag.memory),
                "q": [round(float(v), 4) for v in q[0]],
                "frame": jpg_b64(self.env.unwrapped.ale.getScreenRGB()),
                "stack": png_b64(np.concatenate(list(st), axis=1)),
                "conv1": png_b64(c1), "conv2": png_b64(c2), "conv3": png_b64(c3), "top_conv1": top1,
                "dense_active": int((dense > 0).sum()), "dense_max": round(float(dense.max()), 3),
                "synced": bool(synced), "timing": self.timing(),
            }
            if details:
                details = dict(details)
                details["s_img"], details["s2_img"] = png_b64(details["s_img"]), png_b64(details["s2_img"])
                payload["learn"] = details

        if ended:
            if self.mode == "train":
                ag.end_episode(self.ep_reward)
            self.history.append({"mode": self.mode, "reward": self.ep_reward, "frame": ag.frame_count})
            self.history = self.history[-500:]
            self.new_episode()
        else:
            self.state = s2
        return payload

    def timing(self):
        c, f = self.agent.cfg, self.agent.frame_count
        return {"update_in": (-f) % c["update_after_actions"] or c["update_after_actions"],
                "target_in": (-f) % c["update_target_network"] or c["update_target_network"]}

    def info(self):
        ag = self.agent
        from .atari_dqn import QNetwork
        return {"type": "info", "mode": self.mode, "cfg": ag.cfg, "web_memory": WEB_MEMORY,
                "frame_count": ag.frame_count, "epsilon": round(ag.epsilon, 5), "episode": ag.episode_count,
                "running_reward": round(ag.running_reward, 3), "memory": len(ag.memory),
                "history": self.history[-300:], "train_rewards": ag.all_rewards[-300:],
                "layers": QNetwork.layer_info(ag.model), "n_params": sum(p.numel() for p in ag.model.parameters()),
                "checkpoint": read_meta(), "running": self.running, "fast": self.fast_running,
                "timing": self.timing(), "message": self.message}

    async def broadcast(self, msg):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    def locked_step(self, render=True):
        with self.lock:
            return self.step(render)

    async def play_loop(self):
        while self.running and self.clients:
            t = time.perf_counter()
            d = await asyncio.to_thread(self.locked_step)
            await self.broadcast(d)
            if d["ended"]:
                await self.broadcast(self.info())
            await asyncio.sleep(max(0.0, 1 / self.fps - (time.perf_counter() - t)))
        self.running = False

    def fast_chunk(self, n):
        with self.lock:
            for _ in range(n):
                if not self.fast_running:
                    break
                self.step(render=False)

    async def fast_train(self, frames):
        """ฝึกแบบ headless (ไม่ส่งภาพ) — เร็วกว่ามาก ส่งความคืบหน้าทุก ๆ 500 frames"""
        self.fast_running, self.mode = True, "train"
        start = self.agent.frame_count
        await self.broadcast(self.info())
        t0 = time.perf_counter()
        while self.fast_running and self.agent.frame_count - start < frames:
            await asyncio.to_thread(self.fast_chunk, min(500, frames - (self.agent.frame_count - start)))
            done = self.agent.frame_count - start
            fps = done / max(1e-6, time.perf_counter() - t0)
            await self.broadcast({"type": "fast_progress", "done": done, "total": frames, "fps": round(fps, 1),
                                  "frame_count": self.agent.frame_count, "epsilon": round(self.agent.epsilon, 5)})
        self.fast_running = False
        self.message = f"ฝึกเสร็จ {self.agent.frame_count - start:,} frames"
        await self.broadcast(self.info())


LAB = None


@router.websocket("/ws/atari")
async def atari_ws(ws: WebSocket):
    global LAB
    await ws.accept()
    if LAB is None:
        await ws.send_json({"type": "loading"})
        LAB = await asyncio.to_thread(AtariLab)
    lab = LAB
    lab.clients.add(ws)
    await ws.send_json(lab.info())
    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            busy = lab.running or lab.fast_running
            if cmd == "step" and not busy:
                d = await asyncio.to_thread(lab.locked_step)
                await lab.broadcast(d)
                if d["ended"]:
                    await lab.broadcast(lab.info())
            elif cmd == "play" and not busy:
                lab.running = True
                lab.task = asyncio.create_task(lab.play_loop())
                await lab.broadcast(lab.info())
            elif cmd == "pause":
                lab.running, lab.fast_running = False, False
                await lab.broadcast(lab.info())
            elif cmd == "fps":
                lab.fps = max(1, min(60, int(msg.get("fps", 30))))
            elif cmd == "mode" and msg.get("mode") in ("train", "greedy", "heuristic", "random") and not lab.fast_running:
                with lab.lock:
                    lab.mode = msg["mode"]
                    lab.new_episode()
                await lab.broadcast(lab.info())
            elif cmd == "fast" and not busy:
                asyncio.create_task(lab.fast_train(max(100, min(2_000_000, int(msg.get("frames", 10000))))))
            elif cmd == "save" and not lab.fast_running:
                MODELS.mkdir(parents=True, exist_ok=True)
                with lab.lock:
                    lab.agent.save(CKPT)
                    save_meta(lab.agent)
                lab.message = f"บันทึก checkpoint แล้ว ({lab.agent.frame_count:,} frames)"
                await lab.broadcast(lab.info())
            elif cmd == "load" and CKPT.exists() and not busy:
                with lab.lock:
                    await asyncio.to_thread(lab.agent.load, CKPT)
                    lab.new_episode()
                lab.message = f"โหลด checkpoint แล้ว ({lab.agent.frame_count:,} frames)"
                await lab.broadcast(lab.info())
            elif cmd == "reset" and not busy:
                with lab.lock:
                    lab.reset_agent()
                lab.message = "สร้าง network ใหม่ (สุ่มน้ำหนัก)"
                await lab.broadcast(lab.info())
    except WebSocketDisconnect:
        pass
    finally:
        lab.clients.discard(ws)
        if not lab.clients:
            lab.running = False
