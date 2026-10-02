"""Maze (Gridworld) + Deep Q-Network เขียนด้วย NumPy ล้วน (ไม่ต้องใช้ TensorFlow)

DQN = Neural Network แทน Q-table + Experience Replay + Target Network
    L(θ) = E_(s,a,r,s')~U(D) [ ( r + γ max_a' Q(s',a'; θ⁻) − Q(s,a; θ) )² ]
"""
import random
from collections import deque

import numpy as np

# '#' = กำแพง, 'S' = จุดเริ่ม, 'G' = เป้าหมาย
MAZES = {
    # maze จากโน้ต (3×6) — เปิดช่อง (2,2) เพราะแผนที่เดิมไม่มีทางไป Goal
    "note": [
        "S.#...",
        "#..#.#",
        "....G.",
    ],
    "medium": [
        "S..#....",
        ".#.#.##.",
        ".#...#..",
        ".####.#.",
        "......#.",
        ".##.#.#.",
        "..#.#...",
        "#...#.#G",
    ],
    "complex": [
        "S...#.....",
        ".##.#.###.",
        ".#..#...#.",
        ".#.###.#..",
        ".#.....#.#",
        ".####.##..",
        "......#..#",
        ".#.##.#.#.",
        ".#..#...#.",
        "...##.#..G",
    ],
}

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 0=ขึ้น 1=ขวา 2=ลง 3=ซ้าย
ACTION_NAMES = ["↑", "→", "↓", "←"]

R_GOAL, R_STEP, R_WALL = 1.0, -0.04, -0.25


class Maze:
    def __init__(self, rows):
        self.grid = [list(r) for r in rows]
        self.h, self.w = len(rows), len(rows[0])
        self.start = self._find("S")
        self.goal = self._find("G")
        self.max_steps = self.h * self.w * 3
        self.reset()

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            if ch in row:
                return (r, row.index(ch))

    def is_wall(self, r, c):
        return not (0 <= r < self.h and 0 <= c < self.w) or self.grid[r][c] == "#"

    def toggle_wall(self, r, c):
        if (r, c) in (self.start, self.goal) or not (0 <= r < self.h and 0 <= c < self.w):
            return
        self.grid[r][c] = "." if self.grid[r][c] == "#" else "#"

    @property
    def n_states(self):
        return self.h * self.w

    def encode(self, pos):
        """State → input ของ network: one-hot ตำแหน่ง (ขนาด h×w)"""
        v = np.zeros(self.n_states, dtype=np.float32)
        v[pos[0] * self.w + pos[1]] = 1.0
        return v

    def reset(self):
        self.pos, self.steps = self.start, 0
        return self.pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        if self.is_wall(r, c):
            reward, bumped = R_WALL, True  # ชนกำแพง → อยู่ที่เดิม
        else:
            self.pos, bumped = (r, c), False
            reward = R_GOAL if self.pos == self.goal else R_STEP
        terminated = self.pos == self.goal
        truncated = not terminated and self.steps >= self.max_steps
        return self.pos, reward, terminated, truncated, bumped

    def free_cells(self):
        return [(r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] != "#"]

    def shortest_path_len(self):
        """BFS เพื่อตรวจว่า maze มีทางไป Goal และความยาวทางที่สั้นที่สุด"""
        q, seen = deque([(self.start, 0)]), {self.start}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == self.goal:
                return d
            for dr, dc in ACTIONS:
                n = (r + dr, c + dc)
                if not self.is_wall(*n) and n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return None


class MLP:
    """Fully-connected network: input → 64 (ReLU) → 64 (ReLU) → 4 (linear) พร้อม Adam optimizer"""

    def __init__(self, sizes, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]

    def forward(self, X, keep=False):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts if keep else acts[-1]

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def train_batch(self, X, actions, y):
        """ลด MSE ระหว่าง Q(s,a;θ) กับ target y เฉพาะ action ที่ทำจริง"""
        acts = self.forward(X, keep=True)
        q = acts[-1]
        idx = np.arange(len(X))
        err = q[idx, actions] - y
        loss = float(np.mean(err ** 2))
        grad = np.zeros_like(q)
        grad[idx, actions] = 2 * np.clip(err, -1, 1) / len(X)  # clip gradient (คล้าย Huber loss)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(axis=0)
            if i:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params, grads = self.W + self.b, gW + gb
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)
        return loss, q[idx, actions]

    @property
    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))


class DQNAgent:
    def __init__(self, n_inputs, n_actions=4, hidden=64, gamma=0.95, lr=1e-3, buffer=5000,
                 batch=32, target_every=100, eps_decay=0.97, seed=0):
        self.online = MLP([n_inputs, hidden, hidden, n_actions], lr, seed)
        self.target = MLP([n_inputs, hidden, hidden, n_actions], lr, seed)
        self.target.copy_from(self.online)
        self.memory = deque(maxlen=buffer)  # Replay Buffer D
        self.gamma, self.batch, self.target_every = gamma, batch, target_every
        self.epsilon, self.eps_min, self.eps_decay = 1.0, 0.05, eps_decay
        self.learn_steps, self.episodes = 0, 0
        self.n_actions = n_actions

    def q_values(self, x):
        return self.online.forward(x[None])[0]

    def act(self, x, explore=True):
        u = random.random()
        if explore and u < self.epsilon:
            return random.randrange(self.n_actions), True, u
        return int(np.argmax(self.q_values(x))), False, u

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        """สุ่ม mini-batch จาก D → คำนวณ target ด้วย θ⁻ → gradient step บน θ
        คืนรายละเอียดของตัวอย่างแรกใน batch ไว้แสดงการแทนค่าในสมการ"""
        if len(self.memory) < self.batch:
            return None
        batch = random.sample(self.memory, self.batch)
        S = np.stack([b[0] for b in batch])
        A = np.array([b[1] for b in batch])
        R = np.array([b[2] for b in batch], dtype=np.float32)
        S2 = np.stack([b[3] for b in batch])
        D = np.array([b[4] for b in batch], dtype=np.float32)
        q_next = self.target.forward(S2)
        max_next = q_next.max(axis=1)
        y = R + self.gamma * (1 - D) * max_next
        loss, q_sa = self.online.train_batch(S, A, y)
        self.learn_steps += 1
        synced = False
        if self.learn_steps % self.target_every == 0:
            self.target.copy_from(self.online)  # θ⁻ ← θ
            synced = True
        return {
            "loss": loss, "synced": synced, "batch": self.batch,
            "sample": {"s": int(np.argmax(S[0])), "a": int(A[0]), "r": round(float(R[0]), 4), "s2": int(np.argmax(S2[0])),
                       "done": bool(D[0]), "q_next": q_next[0].tolist(), "max_next": float(max_next[0]),
                       "y": float(y[0]), "q_sa": float(q_sa[0]), "sq_err": float((y[0] - q_sa[0]) ** 2)},
        }

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


def run_episode(maze, agent, learn=True):
    """เล่น 1 episode แบบ headless คืนค่า (reward รวม, จำนวน step, ถึง goal หรือไม่)"""
    pos, total, done = maze.reset(), 0.0, False
    while not done:
        x = maze.encode(pos)
        a, _, _ = agent.act(x, explore=learn)
        pos, r, term, trunc, _ = maze.step(a)
        if learn:
            agent.remember(x, a, r, maze.encode(pos), term)
            agent.replay()
        total, done = total + r, term or trunc
    if learn:
        agent.end_episode()
    return total, maze.steps, maze.pos == maze.goal
