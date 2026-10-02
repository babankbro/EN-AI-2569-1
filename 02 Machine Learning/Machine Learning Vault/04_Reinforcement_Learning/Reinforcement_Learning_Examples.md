# เจาะลึกการประยุกต์ใช้ Reinforcement Learning พร้อมสมการและโค้ด

ในหน้านี้เราจะมาเจาะลึกกรณีศึกษา (Case Studies) ของ Reinforcement Learning (RL) ใน 3 หัวข้อหลัก โดยจะอธิบายตั้งแต่สมการคณิตศาสตร์ที่อยู่เบื้องหลัง ภาพประกอบ และตัวอย่างโค้ดเพื่อให้เห็นภาพการทำงานจริง

---

## 1. การทรงตัวของไม้บนรถเข็น (CartPole) ด้วย OpenAI Gym

**CartPole** เป็นปัญหาพื้นฐานของ RL สภาพแวดล้อมคือรถเข็นที่เคลื่อนที่บนรางเสียดทานศูนย์ โดยมีท่อนไม้ตั้งอยู่บนรถเข็น เป้าหมายคือการขยับรถเข็นไปทางซ้ายหรือขวาเพื่อไม่ให้ท่อนไม้ล้ม

### สมการที่เกี่ยวข้อง: The Bellman Equation (สำหรับ Q-Learning)
ในการแก้ปัญหานี้ หากเราใช้แนวคิด Q-Learning เราจะต้องประมาณค่า Q-value ซึ่งอธิบายด้วยสมการ Bellman:

$$ Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a') $$

- **$Q(s, a)$**: มูลค่า (Quality) ของการเลือกทำ action $a$ ใน state $s$
- **$R(s, a)$**: รางวัล (Reward) ที่ได้รับทันทีจากการทำ action $a$
- **$\gamma$ (Gamma)**: Discount factor (0 ถึง 1) เป็นตัวกำหนดว่าเราให้ความสำคัญกับรางวัลในอนาคตมากแค่ไหน
- **$\max_{a'} Q(s', a')$**: ค่า Q-value ที่สูงที่สุดใน state ถัดไป ($s'$)

### ภาพประกอบ
```mermaid
graph TD
    A[State ปัจจุบัน: มุมไม้, ตำแหน่งรถ] --> B{Agent ตัดสินใจ (Policy)};
    B -->|Action: ดันซ้าย (0)| C[ท่อนไม้เอียงขวา];
    B -->|Action: ดันขวา (1)| D[ท่อนไม้เอียงซ้าย];
    C --> E[Environment อัปเดต State ถัดไป];
    D --> E;
    E --> F((ได้ Reward +1 หากไม้ยังไม่ล้ม));
    F --> A;
```

### ตัวอย่างโค้ด (การใช้ Deep Q-Network เบื้องต้น)
เนื่องจาก State ของ CartPole เป็นค่าต่อเนื่อง (Continuous) การใช้ Deep Q-Network (DQN) จึงเหมาะสมกว่าตาราง Q-Table ธรรมดา
```python
import gym
import numpy as np
from tensorflow import keras

# 1. สร้าง Environment
env = gym.make('CartPole-v1')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# 2. สร้างโมเดล Neural Network เพื่อประมาณค่า Q-Value
model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(n_states,), activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(n_actions, activation='linear')
])
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))

# 3. การเลือก Action แบบ Epsilon-Greedy
def act(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample() # สุ่มเพื่อสำรวจ (Exploration)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0]) # เลือก Action ที่ Q-Value สูงสุด (Exploitation)

# (ในความเป็นจริงจะต้องมีส่วนของ Replay Memory และโมเดลสำหรับฝึกสอนด้วย)
```

---

## 2. การหาเส้นทางในเขาวงกต (Walk in Maze) ยกระดับด้วย Deep RL (DQN)

อ้างอิงจากบทความของ TWIML AI เรื่อง **Deep Reinforcement Learning** แม้ว่า RL ปกติจะทำได้ดี แต่เมื่อแผนที่เขาวงกตมีขนาดใหญ่มาก การใช้ตาราง Q-Table แบบเดิมจะไม่สามารถเก็บข้อมูลได้พอ (Curse of Dimensionality) และในโลกความเป็นจริง เราอาจไม่รู้กฎทั้งหมดของ Environment ล่วงหน้า การยกระดับปัญหาเขาวงกตจึงใช้ **Deep Q-Network (DQN)** ซึ่งใช้ Neural Network เป็นตัวแทนตาราง

### สมการที่เกี่ยวข้อง: DQN Loss Function และ Experience Replay
เพื่อแก้ปัญหาความไม่เสถียรของ Neural Network สิ่งที่เพิ่มเข้ามาคือ **Experience Replay** (การจำลองประสบการณ์ในอดีตมาสอนโมเดลแบบสุ่ม) โดยใช้สมการ Loss Function:

$$ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

- **$\mathbb{E}_{(s,a,r,s') \sim U(D)}$**: สุ่มหยิบประสบการณ์อดีตจาก Replay Buffer $D$
- **$\theta^-$**: น้ำหนักของ Target Network ที่อัปเดตช้ากว่า เพื่อให้เป้าหมายในการเรียนรู้นิ่งขึ้น

### ภาพประกอบเขาวงกตแบบซับซ้อน (Complex Gridworld)
|   | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 🟢Start | ⬜️ | 🟥 | ⬜️ | ⬜️ | ⬜️ |
| 1 | 🟥 | ⬜️ | ⬜️ | 🟥 | ⬜️ | 🟥 |
| 2 | ⬜️ | ⬜️ | 🟥 | ⬜️ | 🏆Goal | ⬜️ |
*ในเขาวงกตขนาดใหญ่และซับซ้อน State จะถูกแปลงเป็นพิกัดหรือภาพ เข้าสู่ Neural Network โดยตรง*

### ตัวอย่างโค้ด (DQN นำร่องสำหรับ Maze)
```python
import numpy as np
import tensorflow as tf
from collections import deque
import random

# 1. สร้างโมเดล Neural Network แทนตาราง Q-Table
def build_dqn_model(state_size, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=state_size, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

# 2. Replay Buffer (เก็บประสบการณ์ไว้สุ่มเรียนรู้)
memory = deque(maxlen=2000)

def replay(model, batch_size, gamma):
    if len(memory) < batch_size: return
    minibatch = random.sample(memory, batch_size) # สุ่มหยิบประสบการณ์
    
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            # คำนวณค่า Q ล่วงหน้าจาก Next State
            target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0])
        
        # ปรับค่า Q เฉพาะ Action ที่ทำไป
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        
        # อัปเดต Neural Network (ลด Loss Function)
        model.fit(state, target_f, epochs=1, verbose=0)

# ในการเล่นจริง จะต้องนำ state, action, reward, next_state เก็บลง memory 
# แล้วจึงเรียก replay() ทุกๆ step
```

---

## 3. การเล่นเกม Atari (Breakout) ด้วย Deep Q-Network

อ้างอิงจากบทช่วยสอน [Keras: Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout/) ปัญหานี้คือการสอน AI ให้เล่นเกมกระดานเด้งลูกบอลทำลายอิฐ (Breakout) โดยที่ Agent จะไม่รู้กฎของเกมเลย แต่ต้องเรียนรู้จากการดู **"ภาพบนหน้าจอ (Pixels)"** เท่านั้น

### ความท้าทายและการแก้ปัญหา: Frame Stacking และ CNN
ในการเล่นเกมภาพวิดีโอ การดูภาพเพียง 1 เฟรมจะไม่สามารถบอกทิศทางและความเร็วของลูกบอลได้ เราจึงต้องนำภาพ 4 เฟรมล่าสุดมาซ้อนกัน (Frame Stacking) แล้วป้อนเข้าสู่ **Convolutional Neural Network (CNN)** เพื่อให้ AI สกัดคุณลักษณะ (Features) ออกมา

**โครงสร้างของเครือข่ายประสาทเทียม (Deepmind Architecture):**
1. **Input:** ภาพหน้าจอ 4 เฟรมติดกัน ขาวดำ (84x84x4)
2. **Conv2D Layer 1:** 32 ฟิลเตอร์ (8x8) สไตรด์ 4 เพื่อดึงรูปร่างพื้นฐาน
3. **Conv2D Layer 2 & 3:** 64 ฟิลเตอร์ เพื่อดึงรายละเอียดที่ซับซ้อนขึ้น
4. **Dense Layer:** แปลงเป็น 4 ค่า (4 Actions: ขยับซ้าย, ขวา, อยู่เฉยๆ, ปล่อยบอล)

### สมการและหลักการทำงานของ Target Network
ปัญหาหนึ่งของ DQN คือเป้าหมาย ($r + \gamma \max Q(s',a')$) จะขยับไปมาตลอดเวลาเมื่อเราอัปเดตโมเดล เพื่อแก้ปัญหานี้ Keras ใช้ **Target Network ($\theta^-$)** ซึ่งเป็นโมเดลที่ถูกแช่แข็งน้ำหนักไว้ และจะอัปเดตน้ำหนักให้ตรงกับโมเดลหลัก (Prediction Network, $\theta$) แค่ทุกๆ 10,000 Step

$$ L(\theta) = \mathbb{E} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{ใช้ Target Network คงที่}} - \underbrace{Q(s, a; \theta)}_{\text{ใช้ Prediction Network}} \right)^2 \right] $$

### ตัวอย่างโค้ด (การสร้างโมเดลและ Environment ของ Keras)
```python
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

# 1. การเตรียม Environment แบบ Atari
# ใช้ Wrapper เพื่อแปลงภาพเป็นขาวดำ 84x84 และนำภาพ 4 เฟรมมารวมกันเป็น 1 State
env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

num_actions = 4 # จำนวนปุ่มที่กดได้

# 2. การสร้าง CNN สำหรับ Deep Q-Network
def create_q_model():
    return keras.Sequential([
        # สลับแกน (Transpose) ให้ตรงกับฟอร์แมตของ Keras
        layers.Lambda(
            lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
            output_shape=(84, 84, 4),
            input_shape=(4, 84, 84),
        ),
        layers.Conv2D(32, 8, strides=4, activation="relu"),
        layers.Conv2D(64, 4, strides=2, activation="relu"),
        layers.Conv2D(64, 3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear") # ส่งออก Q-Value 4 ค่า
    ])

# 3. กำหนดโมเดลหลักและ Target Network
model = create_q_model()
model_target = create_q_model() # แช่แข็งไว้ใช้คำนวณเป้าหมาย (Target)

# หลังจากนี้จะเป็นการใช้ Epsilon-Greedy ควบคู่กับ Experience Replay (เก็บประวัติ)
# และเรียก model_target.set_weights(model.get_weights()) ทุกๆ 10,000 steps
```
