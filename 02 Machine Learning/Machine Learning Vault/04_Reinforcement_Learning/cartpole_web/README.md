# 🕹️ RL Simulation Lab (Docker)

เว็บแอปสำหรับดู **CartPole-v1** ทำงานแบบ real-time บน browser
Python (Gymnasium) รัน simulation และ agent ฝั่ง server → ส่ง state ผ่าน WebSocket → browser วาดรถเข็นและท่อนไม้บน `<canvas>`

ต่อยอดจากหัวข้อที่ 1 ใน [[Reinforcement_Learning_Examples]]

## วิธีรัน
ต้องเปิด **Docker Desktop** ก่อน แล้วรันในโฟลเดอร์นี้:

```bash
docker compose up -d --build
```

เปิด browser ที่ **http://localhost:8000** (หน้ารวม) · หยุดด้วย `docker compose down`

| หน้า | URL | เนื้อหา |
|---|---|---|
| หน้ารวม | `/` | ลิงก์ทุก simulation + ตารางเปรียบเทียบ Q-Learning vs DQN |
| CartPole | `/cartpole` | เปรียบเทียบ agent + training curve |
| CartPole Q-Table Lab | `/qtable` | Q-Learning ทีละ step + แทนค่าสมการ Bellman |
| Maze DQN | `/maze` | Walk in Maze ด้วย Deep Q-Network (หัวข้อที่ 2 ในโน้ต) |
| Atari DQN | `/atari` | Breakout จาก pixel ตามตัวอย่าง Keras (หัวข้อที่ 3 ในโน้ต) |

## ฝึก Q-Learning ล่วงหน้า (Pretrained)
```bash
python -m app.train --episodes 3000
```
- ฝึก 3000 episodes (~45 วินาที) และทดสอบแบบ greedy ทุก 100 episodes
- เก็บ **checkpoint ที่ดีที่สุด** ไว้ที่ `app/models/qtable.npy` + กราฟการฝึกที่ `app/models/history.json`
- ผลที่ได้: eval score **500/500** และทดสอบซ้ำ 100 episodes ได้ 500 ทุกครั้ง
- หน้าเว็บเลือก **Q-Learning (pretrained, greedy)** เป็นค่าเริ่มต้น และแสดงกราฟ training curve ด้านล่าง
- ข้อสังเกต: กราฟการฝึกไม่เสถียร (คะแนนตกแล้วกลับขึ้น) — เป็นลักษณะปกติของ tabular Q-Learning บน state ที่ถูก discretize จึงต้องเก็บ checkpoint ที่ดีที่สุด
- ถ้าลบ `app/models/` ออก Docker จะฝึกใหม่ให้อัตโนมัติตอน build

หลังฝึกใหม่ ให้รัน `docker compose up -d --build` อีกครั้งเพื่อให้ image ใช้โมเดลล่าสุด

## หน้า Q-Table Lab (http://localhost:8000/qtable)
เดิน Q-Learning **ทีละ step** (กด Step หรือ Play) เพื่อดูทุกขั้นตอนของ 1 update:

1. **State → Discretize** — ค่า x, ẋ, θ, θ̇ ถูกแบ่งเป็น bin → ได้ s = (x_bin, ẋ_bin, θ_bin, θ̇_bin)
2. **ε-greedy** — สุ่ม u ถ้า u < ε → Explore (สุ่ม) ไม่งั้น Exploit (argmax Q) พร้อมแสดง Q(s,←), Q(s,→)
3. **แทนค่าในสมการ Bellman** — แสดง TD target, TD error δ และค่า Q ใหม่ด้วยตัวเลขจริง
4. **Q-table heatmap** — ตาราง θ × θ̇ (ที่ x, ẋ bin คงที่) สี = max Q, ลูกศร = policy, กรอบม่วง = s ที่เพิ่ง update, กรอบประ = s′
5. **Update log** — ประวัติการ update ล่าสุด

เลือกเริ่มจาก Q-table ว่าง (เห็นการเรียนรู้ตั้งแต่ 0) หรือ pretrained (ค่า Q ≈ 90 ใกล้ 1/(1−γ) = 100) และกด "ฝึกเร็ว" เพื่อข้ามไปหลายร้อย episode

## หน้า Maze DQN (http://localhost:8000/maze)
DQN เขียนด้วย **NumPy ล้วน** (`app/maze_dqn.py`) ไม่ต้องใช้ TensorFlow: MLP 2 hidden layers × 64 (ReLU) + Adam

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

1. **Maze** — สีคือ max Q(s,·) จาก network ลูกศรคือ policy เส้นคือทางเดินใน episode · กด ✏️ แก้กำแพงได้
2. **Q-Network** — forward pass: input one-hot → hidden 1 → hidden 2 → Q ของ 4 actions (↑ → ↓ ←)
3. **แทนค่า DQN loss** — สุ่มตัวอย่างจาก mini-batch แสดง max Q(s′; θ⁻) จาก target network, target y, squared error, batch loss และ countdown ก่อน sync θ⁻ ← θ
4. **Replay Buffer D** — ประสบการณ์ล่าสุด (s, a, r, s′, done) และขนาด |D|
5. **กราฟ** — steps ต่อ episode เทียบกับทางสั้นสุด (BFS) และ loss

| Maze | ขนาด | ทางสั้นสุด | ผลการทดสอบ |
|---|---|---|---|
| จากโน้ต | 3×6 | 6 | เจอทางสั้นสุดภายใน 50 episodes |
| Medium | 8×8 | 14 | เจอทางสั้นสุดภายใน 50 episodes |
| Complex | 10×10 | 18 | เจอทางสั้นสุดภายใน 50–80 episodes |

- Reward: ถึง Goal +1 · เดิน −0.04 · ชนกำแพง −0.25 · γ = 0.95 · batch 32 · buffer 5000 · sync θ⁻ ทุก 100 learn steps
- ⚠️ maze 3×6 ในโน้ตเดิม **ไม่มีทางไป Goal** (ช่องที่เดินถึงได้จาก Start ถูกกำแพงล้อม) จึงเปิดช่อง (2,2) ให้
- maze นี้เล็กและ input เป็น one-hot ตาราง Q จึงเล็กกว่า network — ใช้เพื่อแสดง **กลไก** ของ DQN ส่วนข้อได้เปรียบจริงคือเมื่อ state ใหญ่/ต่อเนื่อง (เช่นภาพ Atari)

## หน้า Atari Breakout DQN (http://localhost:8000/atari)
ทำตาม [Keras: Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout/) (กรณีศึกษาที่ 3 ในโน้ต)
เขียนด้วย **PyTorch CPU** (`app/atari_dqn.py`) แต่คง architecture, hyperparameters และลำดับขั้นของ training loop เหมือน Keras ทุกอย่าง

- **Environment:** `BreakoutNoFrameskip-v4` + `AtariPreprocessing` (frame skip 4, grayscale, 84×84) + frame stack 4
- **CNN:** Conv(32, 8×8, s4) → Conv(64, 4×4, s2) → Conv(64, 3×3, s1) → Dense 512 → Dense 4 (1,686,180 params)
- **Loss:** Huber · target `y = r + γ max Q(s′; θ⁻)` และ terminal → `y = −1` (ตาม Keras) · Adam lr 0.00025, clipnorm 1.0
- **ε-greedy:** สุ่มล้วน 50,000 frames แรก แล้วลดเชิงเส้นจนถึง 0.1 ที่ 1M frames · train ทุก 4 frames · sync target ทุก 10,000 frames

หน้าเว็บแสดง: ภาพเกม, input 4 เฟรม 84×84, feature maps ของ conv ทั้ง 3 ชั้น, Q ของ 4 actions, การแทนค่า Huber loss จาก mini-batch จริง, กราฟ ε schedule และ reward
มี 4 โหมด: **DQN Train**, **DQN เล่นด้วย network** (ε = 0.05), **Heuristic** (ขยับ paddle ตามลูกบอลจากสีในภาพ ได้ประมาณ 30–120 คะแนน) และ **Random** (1–4 คะแนน)

### ⚠️ เวลาที่ต้องใช้ฝึก
ตัวอย่าง Keras ได้ผลดีที่ประมาณ **10 ล้าน frames** (ไม่ถึง 24 ชั่วโมงบนเครื่องสมัยใหม่) · บน CPU ของเครื่องนี้ได้ประมาณ 120 frames/s → 10M frames ≈ 23 ชั่วโมง
การกด Train ในหน้าเว็บจึงเห็นแค่ **กลไก** (ช่วง 50,000 frames แรกเป็นการสุ่มล้วน) ถ้าต้องการ agent ที่เก่ง ให้ฝึก offline:

```bash
docker compose exec cartpole python -m app.atari_train --frames 2000000
```
เพิ่ม `--resume` เพื่อฝึกต่อ แล้วกด **📂 Load checkpoint** ในหน้าเว็บ · checkpoint อยู่ที่ `app/models/atari/` (mount ออกมานอก container)

- replay memory: Keras ใช้ 100,000 (≈ 5.6 GB RAM) → หน้าเว็บใช้ 10,000 (`ATARI_MEMORY` ใน docker-compose.yml) · script offline ใช้ 50,000 (`--memory`)
- Keras ป้อน pixel 0–255 โดยไม่ scale → ค่า Q ช่วงแรกสูงเกินจริง (เช่น ~9) ทำตามต้นฉบับไว้
- ต้องเปิดหน้า Atari ครั้งแรกรอสักครู่ (โหลด PyTorch + ALE) · agent ตัวเดียวใช้ร่วมกันทุกแท็บ

## Agents ที่มีให้เลือก
| Policy | หลักการ | ผลโดยประมาณ (reward เฉลี่ย) |
|---|---|---|
| **Random** | สุ่ม action ซ้าย/ขวา (baseline) | ~25 |
| **Heuristic** | PD controller: `u = 0.1x + 0.5ẋ + 10θ + 2θ̇` ดันไปทางที่ไม้เอียง | 500 (เต็ม) |
| **Q-Learning (pretrained)** | โหลด Q-table ที่ฝึกแล้ว เลือก action แบบ greedy (ไม่สุ่ม ไม่เรียนรู้ต่อ) | 500 (เต็ม) |
| **Q-Learning (from scratch)** | Tabular Q-table บน state ที่แบ่งช่อง (3×3×8×12) ใช้ Bellman update | ~200–400 หลังฝึก 500–1000 episodes |

$$ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] $$

- **Learn while playing** — ให้ Q-Learning เรียนรู้ระหว่างแสดงผล (ε-greedy)
- **Fast-train** — ฝึกแบบ headless (ไม่วาดภาพ) หลายร้อย episode ในไม่กี่วินาที แล้วกด Start เพื่อดูผล
- กราฟ **Reward per episode** แสดงจุดของแต่ละ episode และเส้น moving average 20

## โครงสร้างไฟล์
```
cartpole_web/
├── Dockerfile, docker-compose.yml, requirements.txt
└── app/
    ├── main.py        # FastAPI + WebSocket loop (env.step → send_json)
    ├── agents.py      # Random / Heuristic / Q-Learning (save / load / evaluate)
    ├── qlab.py        # WebSocket /ws/qlab สำหรับหน้า Q-Table Lab (step-by-step)
    ├── maze_dqn.py    # Maze environment + DQN (NumPy MLP, replay buffer, target network)
    ├── maze_ws.py     # WebSocket /ws/maze สำหรับหน้า Maze DQN
    ├── atari_dqn.py   # Breakout env + CNN DQN (PyTorch) ตาม Keras
    ├── atari_ws.py    # WebSocket /ws/atari
    ├── atari_train.py # ฝึก Breakout แบบ offline + checkpoint
    ├── train.py       # ฝึก Q-Learning + เก็บ best checkpoint
    ├── models/        # qtable.npy, history.json, atari/breakout_dqn.pt
    └── static/
        ├── hub.html        # หน้ารวม
        ├── cartpole.html   # หน้า CartPole Simulation
        ├── maze.html       # หน้า Maze DQN
        ├── atari.html      # หน้า Atari Breakout DQN
        └── qtable.html     # หน้า Q-Table Lab
```

#reinforcement_learning #cartpole #qlearning #dqn #maze #atari #breakout #docker
