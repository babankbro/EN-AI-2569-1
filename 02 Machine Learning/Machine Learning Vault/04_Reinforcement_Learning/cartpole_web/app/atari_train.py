"""ฝึก DQN บน Breakout แบบ offline (ตาม training loop ของตัวอย่าง Keras) แล้วบันทึก checkpoint ให้หน้าเว็บโหลด

รันใน container ที่กำลังทำงานอยู่:
    docker compose exec cartpole python -m app.atari_train --frames 2000000
ต่อจาก checkpoint เดิม:
    docker compose exec cartpole python -m app.atari_train --frames 2000000 --resume

ตัวอย่าง Keras: ได้ผลดีที่ประมาณ 10 ล้าน frames และถือว่า "solved" เมื่อ running reward (100 episodes) > 40
"""
import argparse
import time

from .atari_dqn import AtariDQN, make_env
from .atari_ws import CKPT, MODELS, save_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=1_000_000)
    p.add_argument("--memory", type=int, default=50000, help="Keras ใช้ 100000 (~5.6 GB RAM)")
    p.add_argument("--save-every", type=int, default=50000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    agent = AtariDQN(memory_size=args.memory)
    if args.resume and CKPT.exists():
        agent.load(CKPT)
        print(f"resume from frame {agent.frame_count:,}")
    env = make_env()
    target, t0, start = agent.frame_count + args.frames, time.time(), agent.frame_count

    while agent.frame_count < target:
        state, _ = env.reset()
        episode_reward = 0.0
        for _ in range(1, agent.cfg["max_steps_per_episode"]):
            action, _ = agent.act(state, train=True)
            state_next, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            agent.remember(state, action, reward, state_next, done)
            state = state_next
            _, synced = agent.learn()
            if synced:
                fps = (agent.frame_count - start) / (time.time() - t0)
                print(f"running reward: {agent.running_reward:.2f} at episode {agent.episode_count}, "
                      f"frame count {agent.frame_count:,}, epsilon {agent.epsilon:.3f}, {fps:.0f} frames/s", flush=True)
            if agent.frame_count % args.save_every == 0:
                agent.save(CKPT)
                save_meta(agent)
            if done or truncated:
                break
        agent.end_episode(episode_reward)
        if agent.running_reward > 40:
            print(f"Solved at episode {agent.episode_count}!")
            break

    agent.save(CKPT)
    save_meta(agent)
    print(f"saved {CKPT} ({agent.frame_count:,} frames, running reward {agent.running_reward:.2f})")


if __name__ == "__main__":
    main()
