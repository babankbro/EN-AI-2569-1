# The Machine Learning Landscape

สรุปแนวคิดพื้นฐานจากบทที่ 1

## ประเภทของระบบ Machine Learning

1. **Supervised / Unsupervised / Semi-supervised / Reinforcement Learning**
   - การแบ่งตามว่ามี "ผู้สอน" (Labels) หรือไม่

2. **Batch vs Online Learning**
   - **Batch Learning:** ฝึกสอนโมเดลด้วยข้อมูลทั้งหมดที่มีรวดเดียว มักใช้เวลาและทรัพยากรเยอะ (Offline)
   - **Online Learning:** ทยอยฝึกสอนโมเดลทีละนิดด้วยข้อมูลใหม่ (Mini-batches) เหมาะสำหรับระบบที่รับข้อมูลเข้ามาเรื่อยๆ (Incremental learning)

3. **Instance-Based vs Model-Based Learning**
   - **Instance-Based:** เรียนรู้โดยการจำข้อมูลตัวอย่าง (Learn by heart) แล้วเปรียบเทียบข้อมูลใหม่กับข้อมูลเดิม (เช่น k-Nearest Neighbors)
   - **Model-Based:** หาความสัมพันธ์ของข้อมูลเพื่อสร้าง "โมเดล" แล้วใช้โมเดลนั้นทำนายผล

## ความท้าทายหลัก (Main Challenges)

- **ข้อมูลน้อยเกินไป (Insufficient Quantity of Training Data)**
- **ข้อมูลไม่เป็นตัวแทนที่ดี (Nonrepresentative Training Data):** นำไปสู่ Sampling Bias
- **ข้อมูลไม่มีคุณภาพ (Poor-Quality Data):** มี Error, Outlier, Noise เยอะ
- **ฟีเจอร์ไม่เกี่ยวข้อง (Irrelevant Features):** "Garbage in, garbage out" นำไปสู่ความสำคัญของการทำ Feature Engineering (Selection, Extraction)
- **Overfitting:** โมเดลซับซ้อนเกินไป ท่องจำข้อมูลสอนได้ดี แต่ทำนายข้อมูลใหม่แย่ (แก้ไขด้วย Regularization, หาข้อมูลเพิ่ม)
- **Underfitting:** โมเดลเรียบง่ายเกินไป ไม่สามารถจับรูปแบบข้อมูลได้ (แก้ไขด้วยการเพิ่มความซับซ้อนโมเดล, ลด Regularization)

## การประเมินผล (Testing and Validating)

- ปกติเราจะแบ่งข้อมูลเป็น **Training Set** (ส่วนใหญ่) และ **Test Set** (เก็บไว้ทดสอบขั้นสุดท้าย ห้ามแตะ!)
- การปรับจูนพารามิเตอร์ของโมเดล (Hyperparameter Tuning) หากใช้ Test set จะทำให้เกิด Data Snooping Bias ทำให้ประเมินผลเกินจริง
- **วิธีแก้:** ต้องแบ่งข้อมูลมาอีกส่วนเรียกว่า **Validation Set** ไว้ใช้จูนโมเดล หรือใช้วิธี **Cross-Validation** (สลับส่วนข้อมูลมาทำ Validation)
