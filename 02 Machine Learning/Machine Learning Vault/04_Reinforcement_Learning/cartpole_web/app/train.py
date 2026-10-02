"""ฝึก Q-Learning สำหรับ CartPole-v1 แล้วบันทึก Q-table ที่ดีที่สุด

รัน:  python -m app.train --episodes 3000
ผลลัพธ์:
  app/models/qtable.npy        Q-table ที่ได้คะแนน evaluation สูงสุด
  app/models/history.json      reward ทุก episode + ผล evaluation ทุก 100 episode
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .agents import QLearningAgent

MODELS = Path(__file__).parent / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    eval_env = gym.make("CartPole-v1")
    eval_env.reset(seed=args.seed + 1)

    agent = QLearningAgent()
    rewards, evals = [], []
    best_score, best_q, best_ep = -1.0, None, 0

    for done in range(0, args.episodes, args.eval_every):
        rewards += agent.train(env, min(args.eval_every, args.episodes - done))
        score = agent.evaluate(eval_env, 20)
        evals.append({"episode": agent.episodes, "score": score})
        # เก็บ Q-table ที่ดีที่สุด (checkpoint) เพราะ Q-Learning อาจแกว่ง/ลืมได้ระหว่างฝึก
        if score >= best_score:
            best_score, best_q, best_ep = score, agent.q.copy(), agent.episodes
        print(f"episode {agent.episodes:5d} | train avg100 {np.mean(rewards[-100:]):6.1f} "
              f"| eval {score:6.1f} | eps {agent.epsilon:.3f}")

    MODELS.mkdir(exist_ok=True)
    np.save(MODELS / "qtable.npy", best_q)
    (MODELS / "history.json").write_text(json.dumps({
        "rewards": rewards, "evals": evals,
        "best": {"episode": best_ep, "score": best_score}, "episodes": args.episodes,
    }))
    print(f"saved best Q-table from episode {best_ep} (eval score {best_score:.1f})")


if __name__ == "__main__":
    main()
