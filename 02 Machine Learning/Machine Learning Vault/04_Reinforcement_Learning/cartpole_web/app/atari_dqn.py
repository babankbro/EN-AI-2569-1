"""Deep Q-Learning for Atari Breakout — ตามตัวอย่าง Keras
https://keras.io/examples/rl/deep_q_network_breakout/

เขียนด้วย PyTorch (CPU) แต่คง architecture / hyperparameters / ลำดับขั้นตอนของตัวอย่าง Keras ไว้ทั้งหมด
(Keras ต้องใช้ TensorFlow ซึ่งทำให้ Docker image ใหญ่มาก)
"""
import random

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

# hyperparameters เหมือนตัวอย่าง Keras ทุกค่า
CFG = {
    "seed": 42,
    "gamma": 0.99,
    "epsilon_min": 0.1,
    "epsilon_max": 1.0,
    "batch_size": 32,
    "max_steps_per_episode": 10000,
    "epsilon_random_frames": 50000,
    "epsilon_greedy_frames": 1_000_000,
    "max_memory_length": 100000,
    "update_after_actions": 4,
    "update_target_network": 10000,
    "learning_rate": 0.00025,
    "clipnorm": 1.0,
}
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT"]


def make_env():
    """env = gym.make("BreakoutNoFrameskip-v4"); AtariPreprocessing(env); FrameStack(env, 4)"""
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env)  # noop 30, frame skip 4, grayscale, resize 84×84
    env = FrameStackObservation(env, 4)  # ซ้อน 4 เฟรมล่าสุด → (4, 84, 84)
    return env


class QNetwork(nn.Module):
    """เทียบกับ create_q_model() ใน Keras:
        Conv2D(32, 8, strides=4, relu) → Conv2D(64, 4, strides=2, relu) → Conv2D(64, 3, strides=1, relu)
        → Flatten → Dense(512, relu) → Dense(4, linear)
    PyTorch ใช้ channels-first (4, 84, 84) อยู่แล้ว จึงไม่ต้องมี Lambda transpose
    """

    def __init__(self, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)
        for m in self.modules():  # Keras default: glorot_uniform + bias = 0
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, keep=False):
        # ตัวอย่าง Keras ป้อนค่า pixel 0–255 ตรง ๆ (ไม่ได้หาร 255)
        x = x.float()
        c1 = torch.relu(self.conv1(x))
        c2 = torch.relu(self.conv2(c1))
        c3 = torch.relu(self.conv3(c2))
        h = torch.relu(self.fc(c3.flatten(1)))
        q = self.out(h)
        return (q, {"conv1": c1, "conv2": c2, "conv3": c3, "dense": h}) if keep else q

    def layer_info(self):
        rows = [("Input", "4 × 84 × 84", 0)]
        shapes = ["32 × 20 × 20", "64 × 9 × 9", "64 × 7 × 7", "512", "4"]
        names = ["Conv2D(32, 8×8, stride 4) + ReLU", "Conv2D(64, 4×4, stride 2) + ReLU",
                 "Conv2D(64, 3×3, stride 1) + ReLU", "Flatten 3136 → Dense(512) + ReLU", "Dense(4) linear → Q(s,a)"]
        for name, shape, layer in zip(names, shapes, [self.conv1, self.conv2, self.conv3, self.fc, self.out]):
            rows.append((name, shape, sum(p.numel() for p in layer.parameters())))
        return rows


class ReplayMemory:
    """เก็บ (state, action, reward, state_next, done) เป็น uint8 ใน ring buffer
    (Keras ใช้ list หลายอันแล้ว del ตัวเก่าสุดเมื่อเกิน max_memory_length)"""

    def __init__(self, size):
        self.size, self.n, self.i = size, 0, 0
        self.s = np.zeros((size, 4, 84, 84), np.uint8)
        self.s2 = np.zeros((size, 4, 84, 84), np.uint8)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k):
        idx = np.random.choice(self.n, size=k)  # เหมือน np.random.choice(range(len(done_history)), size=batch_size)
        return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


class AtariDQN:
    def __init__(self, memory_size=None, cfg=CFG):
        self.cfg = dict(cfg)
        if memory_size:
            self.cfg["max_memory_length"] = memory_size
        torch.manual_seed(self.cfg["seed"])
        self.model, self.model_target = QNetwork(), QNetwork()
        self.model_target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["learning_rate"])
        self.loss_function = nn.HuberLoss()  # keras.losses.Huber() (delta = 1, เฉลี่ยทั้ง batch)
        self.memory = ReplayMemory(self.cfg["max_memory_length"])
        self.epsilon = self.cfg["epsilon_max"]
        self.frame_count = 0
        self.episode_count = 0
        self.episode_reward_history = []  # 100 episode ล่าสุด → running reward
        self.all_rewards = []

    @property
    def running_reward(self):
        h = self.episode_reward_history
        return float(np.mean(h)) if h else 0.0

    def q_values(self, state, keep=False):
        with torch.no_grad():
            return self.model(torch.from_numpy(np.asarray(state))[None], keep=keep)

    def act(self, state, train=True, eval_epsilon=0.05):
        """ε-greedy ตาม Keras: สุ่มทั้งหมดใน 50,000 frames แรก แล้วค่อยลด ε เชิงเส้นจนถึง 0.1 ที่ 1M frames"""
        c = self.cfg
        if train:
            self.frame_count += 1
            if self.frame_count < c["epsilon_random_frames"]:
                action, reason = random.randrange(4), "random_frames"
            elif self.epsilon > np.random.rand():
                action, reason = random.randrange(4), "epsilon"
            else:
                action, reason = int(self.q_values(state).argmax()), "greedy"
            self.epsilon -= (c["epsilon_max"] - c["epsilon_min"]) / c["epsilon_greedy_frames"]
            self.epsilon = max(self.epsilon, c["epsilon_min"])
        elif np.random.rand() < eval_epsilon:
            action, reason = random.randrange(4), "epsilon"
        else:
            action, reason = int(self.q_values(state).argmax()), "greedy"
        return action, reason

    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)

    def learn(self):
        """ทำตาม Keras: ทุก 4 frames สุ่ม batch 32 → target จาก model_target → Huber loss → Adam (clipnorm 1.0)
        คืนค่ารายละเอียดตัวอย่างแรกใน batch สำหรับแสดงการแทนค่าในสมการ"""
        c = self.cfg
        details, synced = None, False
        if self.frame_count % c["update_after_actions"] == 0 and len(self.memory) > c["batch_size"]:
            idx, S, A, R, S2, D = self.memory.sample(c["batch_size"])
            with torch.no_grad():
                future_rewards = self.model_target(torch.from_numpy(S2))
            max_future = future_rewards.max(dim=1).values
            R_t, D_t = torch.from_numpy(R), torch.from_numpy(D)
            updated_q_values = R_t + c["gamma"] * max_future
            # Keras: ถ้าเป็น terminal state ให้ target = −1
            updated_q_values = updated_q_values * (1 - D_t) - D_t

            masks = torch.nn.functional.one_hot(torch.from_numpy(A), 4).float()
            q_values = self.model(torch.from_numpy(S))
            q_action = (q_values * masks).sum(dim=1)
            loss = self.loss_function(q_action, updated_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clipnorm"]))
            self.optimizer.step()

            e = float(updated_q_values[0] - q_action[0].detach())
            details = {
                "index": int(idx[0]), "a": int(A[0]), "r": float(R[0]), "done": bool(D[0]),
                "future": [round(float(v), 4) for v in future_rewards[0]], "max_future": float(max_future[0]),
                "y": float(updated_q_values[0]), "q_all": [round(float(v), 4) for v in q_values[0].detach()],
                "q_sa": float(q_action[0].detach()), "error": e,
                "huber": 0.5 * e * e if abs(e) <= 1 else abs(e) - 0.5,
                "loss": float(loss.detach()), "grad_norm": grad_norm,
                "s_img": S[0][-1], "s2_img": S2[0][-1],
            }
        if self.frame_count and self.frame_count % c["update_target_network"] == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            synced = True
        return details, synced

    def end_episode(self, episode_reward):
        self.episode_reward_history.append(episode_reward)
        if len(self.episode_reward_history) > 100:
            del self.episode_reward_history[:1]
        self.all_rewards.append(episode_reward)
        self.episode_count += 1

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "target": self.model_target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "frame_count": self.frame_count,
                    "epsilon": self.epsilon, "episode_count": self.episode_count,
                    "episode_reward_history": self.episode_reward_history, "all_rewards": self.all_rewards}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model_target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        for k in ("frame_count", "epsilon", "episode_count", "episode_reward_history", "all_rewards"):
            setattr(self, k, ck[k])


def heuristic_action(rgb):
    """Baseline ที่ไม่ได้เรียนรู้: หาตำแหน่งลูกบอลและ paddle จากสีในภาพ แล้วขยับ paddle ตามลูก"""
    red = (rgb[..., 0] == 200) & (rgb[..., 1] == 72) & (rgb[..., 2] == 72)
    ball = np.argwhere(red[93:189, 8:152])  # พื้นที่ระหว่างแถวอิฐกับ paddle
    paddle = np.argwhere(red[189:194, 8:152])
    if len(ball) == 0:
        return 1  # FIRE เพื่อปล่อยลูกใหม่
    bx = ball[:, 1].mean()
    px = paddle[:, 1].mean() if len(paddle) else 72
    if bx > px + 3:
        return 2
    if bx < px - 3:
        return 3
    return 0


def feature_grid(t, cols):
    """รวม feature maps (C, H, W) เป็นภาพเดียว (grid) โดย normalize แต่ละ map เป็น 0–255"""
    a = t.detach().numpy()
    c, h, w = a.shape
    mx = a.reshape(c, -1).max(axis=1)[:, None, None]
    a = np.where(mx > 0, a / np.maximum(mx, 1e-6), 0) * 255
    rows = int(np.ceil(c / cols))
    grid = np.full((rows * (h + 1) - 1, cols * (w + 1) - 1), 40, np.uint8)
    for i in range(c):
        r, k = divmod(i, cols)
        grid[r * (h + 1):r * (h + 1) + h, k * (w + 1):k * (w + 1) + w] = a[i]
    return grid, [int(i) for i in np.argsort(-a.reshape(c, -1).mean(axis=1))[:3]]


def png_b64(img):
    import base64
    return base64.b64encode(cv2.imencode(".png", img)[1]).decode()


def jpg_b64(rgb, q=80):
    import base64
    return base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()
