"""Agents for CartPole-v1.

- RandomAgent     : สุ่ม action (baseline)
- HeuristicAgent  : กฎง่าย ๆ จากฟิสิกส์ (ดันไปทางที่ไม้เอียง)
- QLearningAgent  : Tabular Q-Learning บน state ที่แบ่งเป็นช่อง (discretized)
                    ใช้ Bellman update: Q(s,a) <- Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
"""
import math

import numpy as np


class RandomAgent:
    def act(self, obs, env, explore=True):
        return int(env.action_space.sample())


class HeuristicAgent:
    """ดันรถไปทางเดียวกับที่ไม้เอียง (รวมความเร็วเชิงมุมและตำแหน่งรถ) — คล้าย PD controller"""

    def act(self, obs, env, explore=True):
        x, x_dot, theta, theta_dot = obs
        u = 0.1 * x + 0.5 * x_dot + 10.0 * theta + 2.0 * theta_dot
        return 1 if u > 0 else 0


class QLearningAgent:
    # จำนวนช่องของแต่ละ state: (x, x_dot, theta, theta_dot)
    BINS = np.array([3, 3, 8, 12])
    LOWS = np.array([-2.4, -3.0, -0.21, -3.5])
    HIGHS = np.array([2.4, 3.0, 0.21, 3.5])

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.q = np.zeros(tuple(self.BINS) + (2,))
        self.episodes = 0

    # ε และ α ลดลงตามจำนวน episode ที่เรียนไปแล้ว (explore มากตอนแรก แล้วค่อย exploit)
    @property
    def epsilon(self):
        return max(0.01, min(1.0, 1.0 - math.log10((self.episodes + 1) / 25)))

    @property
    def alpha(self):
        return max(0.1, min(0.5, 1.0 - math.log10((self.episodes + 1) / 25)))

    def discretize(self, obs):
        ratios = (np.clip(obs, self.LOWS, self.HIGHS) - self.LOWS) / (self.HIGHS - self.LOWS)
        idx = np.minimum((ratios * self.BINS).astype(int), self.BINS - 1)
        return tuple(idx)

    def act(self, obs, env, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return int(env.action_space.sample())  # Exploration
        return int(np.argmax(self.q[self.discretize(obs)]))  # Exploitation

    def learn(self, obs, action, reward, next_obs, terminated):
        s, s2 = self.discretize(obs), self.discretize(next_obs)
        target = reward if terminated else reward + self.gamma * np.max(self.q[s2])
        self.q[s + (action,)] += self.alpha * (target - self.q[s + (action,)])

    def save(self, path):
        np.save(path, self.q)

    @classmethod
    def load(cls, path):
        agent = cls()
        agent.q = np.load(path)
        agent.episodes = 10**6  # โมเดลที่ฝึกแล้ว → ε ต่ำสุด
        return agent

    def evaluate(self, env, n_episodes=20):
        """ทดสอบแบบ greedy (ไม่สุ่ม ไม่เรียนรู้) คืนค่า reward เฉลี่ย"""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total, done = 0.0, False
            while not done:
                obs, r, terminated, truncated, _ = env.step(self.act(obs, env, explore=False))
                total, done = total + r, terminated or truncated
            rewards.append(total)
        return float(np.mean(rewards))

    def train(self, env, n_episodes):
        """ฝึกแบบไม่แสดงผล (headless) คืนค่า reward รวมของแต่ละ episode"""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total, done = 0.0, False
            while not done:
                a = self.act(obs, env)
                next_obs, r, terminated, truncated, _ = env.step(a)
                self.learn(obs, a, r, next_obs, terminated)
                obs, total, done = next_obs, total + r, terminated or truncated
            self.episodes += 1
            rewards.append(total)
        return rewards
