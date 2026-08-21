# ตัวอย่างการประยุกต์ใช้งาน Reinforcement Learning (RL)

หน้านี้จะอธิบายถึงตัวอย่างงาน 4 รูปแบบที่สามารถนำแนวคิดของ Reinforcement Learning (RL) ไปประยุกต์ใช้ พร้อมกับอธิบายขั้นตอนการทำงานและตัวอย่างโค้ดเพื่อให้เห็นภาพชัดเจนขึ้น

---

## 1. การทรงตัวของไม้บนรถเข็น (CartPole) ด้วย OpenAI Gym
**โจทย์:** ฝึกสอนโมเดลให้ขยับรถเข็น (ซ้าย/ขวา) เพื่อเลี้ยงให้ท่อนไม้ที่ตั้งอยู่บนรถเข็นไม่ล้มลงมา
นี่คือโจทย์คลาสสิกที่สุดในการเริ่มต้นเรียนรู้ RL (เปรียบเหมือน Hello World ของสายนี้)

- **Environment:** คลาส `CartPole-v1` จากไลบรารี `gym`
- **State:** ข้อมูล 4 ค่า ได้แก่ ตำแหน่งรถเข็น, ความเร็วรถเข็น, มุมของไม้, และความเร็วเชิงมุมของไม้
- **Action:** 0 (ดันรถไปทางซ้าย) หรือ 1 (ดันรถไปทางขวา)
- **Reward:** ได้คะแนน +1 ทุกๆ Step ที่ไม้ยังไม่ล้ม

**ขั้นตอนการทำงานพื้นฐาน:**
1. สร้าง Environment ขึ้นมา
2. สังเกต State ปัจจุบัน
3. ตัดสินใจเลือก Action ตาม Policy (ในตัวอย่างจะใช้แบบสุ่มก่อน)
4. รับค่า State ใหม่ และ Reward จาก Environment แล้ววนซ้ำ

**ตัวอย่างโค้ด (Random Policy):**
```python
import gym

# 1. สร้างสภาพแวดล้อม CartPole
env = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()

for step in range(200):
    env.render() # แสดงผลภาพหน้าจอ
    
    # 2-3. เลือก Action แบบสุ่ม (ซ้ายหรือขวา)
    action = env.action_space.sample()
    
    # 4. ส่ง Action กลับไปที่สภาพแวดล้อมและรับผลลัพธ์
    next_state, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        print(f"จบเกมในรอบที่ {step+1}")
        break

env.close()
```

---

## 2. การหาเส้นทางในเขาวงกต (Gridworld/FrozenLake) ด้วย Q-Learning
**โจทย์:** สอนให้ Agent เดินบนพื้นน้ำแข็ง (Grid 4x4) ไปหาของขวัญ โดยต้องหลบหลุมน้ำแข็งให้ได้

- **State:** ตำแหน่งช่องตารางที่ยืนอยู่ (0 ถึง 15)
- **Action:** เดินขึ้น, ลง, ซ้าย, ขวา (0, 1, 2, 3)
- **Reward:** ได้ +1 เมื่อถึงเป้าหมาย นอกนั้นได้ 0 (ตกหลุมเกมจบ)

**ขั้นตอนการทำงาน (Q-Learning):**
1. สร้าง Q-Table ขนาด [จำนวน State, จำนวน Action] โดยให้ค่าเริ่มต้นเป็น 0
2. เลือก Action โดยใช้เทคนิค $\epsilon$-greedy (สุ่มสำรวจบ้าง เลือกค่าสูงสุดบ้าง)
3. พอเดินไปแล้ว ให้อัปเดตตาราง Q-Table ด้วยสมการ **Bellman Equation**

**ตัวอย่างโค้ด (หลักการอัปเดตตาราง):**
```python
import numpy as np

# กำหนดตาราง Q-Table (16 states, 4 actions)
q_table = np.zeros((16, 4))

learning_rate = 0.8
discount_factor = 0.95 # แกมม่า (Gamma) ให้ความสำคัญกับรางวัลในอนาคต

# ฟังก์ชันอัปเดต Q-Value เมื่อผ่านไป 1 ก้าว
def update_q_table(state, action, reward, next_state):
    # ค่า Q เดิม
    old_value = q_table[state, action]
    # คาดการณ์ผลตอบแทนสูงสุดในก้าวถัดไป
    next_max = np.max(q_table[next_state])
    
    # สมการ Q-Learning (Bellman Equation)
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    q_table[state, action] = new_value

# หมายเหตุ: ในการ Train จริง ต้องวนลูปเล่นเกมหลายๆ รอบ (Episodes)
```

---

## 3. การเล่นเกม Atari (เช่น Breakout) ด้วย Deep Q-Network (DQN)
**โจทย์:** สอน AI ให้เล่นเกมยิงลูกบอลทำลายอิฐ (Breakout) โดยใช้ภาพหน้าจอเกมเป็น Input โดยตรง

- **State:** ภาพพิกเซลของหน้าจอเกมที่ถูกลดสเกลและแปลงเป็นสีเทา (Grayscale)
- **Action:** เลื่อนแป้นซ้าย, ขวา, หรืออยู่นิ่ง
- **Reward:** คะแนนที่ได้จากการทำลายบล็อกอิฐ

**ขั้นตอนการทำงาน (DQN):**
1. ใช้ **Convolutional Neural Network (CNN)** มารับภาพ State แล้วทำนายค่า Q-Value ของทุก Action แทนที่จะใช้ตาราง (เพราะภาพมี State เป็นล้านๆ แบบ)
2. เก็บประสบการณ์การเล่น (State, Action, Reward, Next_State) ไว้ใน **Replay Buffer**
3. สุ่มหยิบประสบการณ์จาก Buffer มาสอน CNN เพื่อลดความเอนเอียงของข้อมูล

**โครงสร้างโค้ดแนวคิด (Keras):**
```python
from tensorflow import keras

# สร้างโมเดล CNN สำหรับเป็น Q-Network
def build_dqn(input_shape, n_actions):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, 8, strides=4, activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
        keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(n_actions) # Output เท่ากับจำนวนปุ่มบังคับในเกม
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-3))
    return model

# output ของโมเดลนี้คือ ค่า Q-Value ของแต่ละ Action 
# จากนั้นเราจะเลือก Action ที่ให้ค่าสูงสุดเพื่อไปบังคับเกม
```

---

## 4. บอทเทรดหุ้นอัตโนมัติ (Automated Stock Trading Agent)
**โจทย์:** สร้าง Agent ที่สามารถตัดสินใจ ซื้อ, ขาย, หรือ ถือ หุ้นเพื่อทำกำไรสูงสุดในตลาดจำลอง

- **State:** ข้อมูลตลาด ณ เวลานั้น เช่น ราคาปิด, Volume, เส้นค่าเฉลี่ย (SMA), MACD หรืออินดิเคเตอร์อื่นๆ ของหุ้นตัวนั้นๆ รวมถึงจำนวนเงินสดและหุ้นที่ถืออยู่
- **Action:** 0 (Hold - ถือ), 1 (Buy - ซื้อ), 2 (Sell - ขาย)
- **Reward:** การเปลี่ยนแปลงของมูลค่าพอร์ตการลงทุน (Portfolio Value) เทียบกับรอบที่แล้ว หรือเปรียบเทียบกำไรสุทธิเมื่อสิ้นสุดรอบการจำลอง

**ขั้นตอนการทำงาน:**
1. สร้าง Custom Environment โดยใช้ข้อมูลราคาหุ้นจริงในอดีต (Historical Data) 
2. Agent จะอ่านค่ากราฟและอินดิเคเตอร์เป็น State
3. Agent ทำการซื้อขาย และได้รับ Reward ตามกำไรหรือขาดทุนที่เกิดขึ้นจริง
4. ค่อยๆ ปรับ Policy ให้ Agent รู้ว่าแพทเทิร์นกราฟแบบไหนควรซื้อ แบบไหนควรขาย

**ตัวอย่างโค้ด (การจำลอง Step ใน Environment สไตล์ Gym):**
```python
class StockTradingEnv:
    def __init__(self, stock_data):
        self.stock_data = stock_data # DataFrame ของราคาหุ้น
        self.current_step = 0
        self.balance = 100000 # เงินตั้งต้น
        self.shares_held = 0
        
    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]['Close']
        
        # 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price
            
        # คำนวณมูลค่าพอร์ตรวม
        net_worth = self.balance + (self.shares_held * current_price)
        
        # ก้าวต่อไป
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        
        # ในที่นี้ใช้มูลค่าพอร์ตเป็นรางวัล (หรืออาจใช้ส่วนต่างความกำไร)
        reward = net_worth 
        next_state = self.stock_data.iloc[self.current_step].values
        
        return next_state, reward, done
```
