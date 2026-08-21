# Reinforcement Learning (RL)

(สรุปความรู้จากหนังสือ Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow บทที่ 18)

Reinforcement Learning (RL) เป็นสาขาหนึ่งของ Machine Learning ที่เกี่ยวกับการสอนให้ **Agent (ตัวกระทำ)** เรียนรู้วิธีการตัดสินใจใน **Environment (สภาพแวดล้อม)** เพื่อให้ได้ **Reward (รางวัล)** รวมสูงสุดเมื่อเวลาผ่านไป

## 🔑 องค์ประกอบสำคัญ (Core Concepts)
1. **Agent:** ตัวตัดสินใจ (เช่น หุ่นยนต์, ตัวละครในเกม, หรือระบบเทรดหุ้น)
2. **Environment:** โลกที่ Agent อาศัยอยู่ ซึ่งจะให้สถานะ (State) และรางวัล (Reward) กลับมา
3. **State ($S$):** สถานการณ์หรือข้อมูลที่ Agent ได้รับจาก Environment ในขณะนั้น
4. **Action ($A$):** การกระทำที่ Agent สามารถทำได้ใน State ปัจจุบัน
5. **Reward ($R$):** ผลตอบแทนจากการกระทำ (อาจเป็นบวกหรือลบก็ได้)
6. **Policy ($\pi$):** "นโยบาย" หรือกฎที่ Agent ใช้ในการเลือก Action เมื่ออยู่ใน State หนึ่งๆ เป้าหมายของ RL คือการหา Optimal Policy ($\pi^*$)

## 🧠 กระบวนการเรียนรู้ (Learning Processes)

### 1. Markov Decision Processes (MDPs)
- ทฤษฎีพื้นฐานของ RL อาศัยสมมติฐานของ Markov: "สถานะในอนาคตขึ้นอยู่กับสถานะและการกระทำปัจจุบันเท่านั้น ไม่ขึ้นอยู่กับอดีตที่ผ่านมา"
- ประกอบด้วย:
  - Transition Probabilities: โอกาสที่จะเปลี่ยนจาก State $s$ ไป State $s'$ เมื่อทำ Action $a$
  - Expected Rewards: รางวัลคาดหวังเมื่อเปลี่ยน State

### 2. Q-Learning
- เป็นอัลกอริทึมที่ Agent จะเรียนรู้ **Q-Values (Quality Values)** ซึ่งคือผลตอบแทนรวมสูงสุดที่คาดว่าจะได้รับในอนาคต หากเริ่มต้นที่ State $s$ และทำ Action $a$
- **Bellman Equation:** เป็นสมการที่ใช้อัปเดตค่า Q-Value โดยการดูจากรางวัลปัจจุบันบวกกับ Q-Value สูงสุดของ State ถัดไป (ที่มีการคิดส่วนลด หรือ Discount factor $\gamma$)
- **Exploration vs. Exploitation:**
  - *Exploration (สำรวจ):* ลองทำอะไรใหม่ๆ เพื่อเรียนรู้ Environment
  - *Exploitation (ใช้ประโยชน์):* เลือก Action ที่ให้ Q-Value สูงสุดตามที่เคยเรียนรู้มา
  - มักใช้เทคนิค $\epsilon$-greedy policy เพื่อรักษาสมดุล (สุ่ม Action ด้วยความน่าจะเป็น $\epsilon$ และเลือก Action ที่ดีที่สุดด้วยความน่าจะเป็น $1-\epsilon$)

### 3. Deep Q-Networks (DQN)
- เมื่อ State มีจำนวนมหาศาล (เช่น พิกเซลบนหน้าจอเกม) การสร้างตาราง Q-Table แบบเดิมจะไม่สามารถทำได้
- **DQN** แก้ปัญหานี้โดยใช้ **Deep Neural Networks** มาช่วยประมาณค่า Q-Value แทนตาราง
- **เทคนิคสำคัญใน DQN:**
  - **Replay Memory:** เก็บประวัติประสบการณ์ (State, Action, Reward, Next State) ไว้ แล้วสุ่มหยิบมาใช้ Train โมเดล เพื่อลดความสัมพันธ์ต่อเนื่องของข้อมูล (Correlation)
  - **Target Network:** ใช้ Network อีกตัวที่อัปเดตช้ากว่ามาช่วยคำนวณ Target Q-Value ทำให้การ Train นิ่งขึ้น (Stabilization)

### 4. Policy Gradients (PG)
- แทนที่จะเรียนรู้ Q-Value อัลกอริทึมกลุ่มนี้จะพยายามปรับปรุง **Policy (พารามิเตอร์ของ Neural Network)** โดยตรงเพื่อให้ได้ Reward สูงสุด
- **REINFORCE Algorithm:**
  1. ให้ Agent ทดลองเล่นเกมจนจบ (1 Episode) โดยใช้ Policy ปัจจุบัน
  2. คำนวณผลตอบแทนรวม (Return) ของแต่ละก้าว
  3. ปรับพารามิเตอร์ (Gradient Ascent) เพื่อเพิ่มโอกาสในการเลือก Action ที่ให้ผลตอบแทนดี และลดโอกาสของ Action ที่ให้ผลตอบแทนแย่

## 🛠️ เครื่องมือที่นิยมใช้
- **OpenAI Gym:** ไลบรารีสำหรับจำลอง Environment ต่างๆ (เช่น CartPole, เกม Atari) เพื่อใช้ฝึกสอนและทดสอบอัลกอริทึม RL
- **TF-Agents:** ไลบรารีของ TensorFlow ที่มีอัลกอริทึม RL มาตรฐานให้ใช้งานได้อย่างมีประสิทธิภาพ
