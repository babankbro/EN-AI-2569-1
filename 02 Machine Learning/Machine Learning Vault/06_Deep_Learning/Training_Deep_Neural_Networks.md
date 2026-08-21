# Training Deep Neural Networks

จากบทที่ 11 Training Deep Neural Networks

เมื่อ Neural Network ลึกขึ้น เราจะเจอปัญหาและต้องหาทางแก้ดังนี้:

## 1. ปัญหา Vanishing / Exploding Gradients
- การทำ Backpropagation จะส่ง Gradient (ความชัน) ย้อนกลับไปปรับค่าน้ำหนัก แต่พอย้อนกลับไปชั้นลึกๆ Gradient อาจจะ **หายไป (Vanishing)** หรือ **ระเบิดใหญ่เกินไป (Exploding)** ทำให้ชั้นล่างๆ เรียนรู้ไม่ได้
- **วิธีแก้:**
  - **Initialization:** กำหนดค่าเริ่มต้นของ Weight ให้เหมาะสม (เช่น Glorot, He, LeCun)
  - **Activation Functions:** ใช้ ReLU และผองเพื่อน (Leaky ReLU, ELU, SELU) แทน Sigmoid/Tanh
  - **Batch Normalization (BN):** ทำการ Normalize ข้อมูลให้อยู่ในสเกลเดียวกัน *ก่อน* หรือ *หลัง* เข้า Activation function ในแต่ละชั้น ช่วยลดปัญหาลงได้มาก
  - **Gradient Clipping:** ป้องกัน Exploding Gradient โดยกำหนดขีดจำกัดความชัน

## 2. Faster Optimizers
เปลี่ยนการเดินลงเขา (Gradient Descent) ให้เร็วขึ้น:
- **Momentum Optimization:** จำความเร็วเดิมไว้ เหมือนลูกโบว์ลิ่งกลิ้งลงเขา ยิ่งกลิ้งยิ่งเร็ว
- **Nesterov Accelerated Gradient (NAG):** เหมือน Momentum แต่มองไปข้างหน้า 1 ก้าว ทำให้แม่นยำกว่า
- **AdaGrad:** ปรับ Learning Rate ให้ช้าลงในแกนที่ชันมาก (มักหยุดเร็วเกินไปใน Deep Learning)
- **RMSProp:** แก้ไข AdaGrad ให้สนเฉพาะ Gradient ล่าสุด ไม่ใช่ทั้งหมด
- **Adam / Nadam:** **ตัวยอดฮิต** เอาข้อดีของ Momentum (จำค่าเฉลี่ย Gradient) และ RMSProp (จำค่าเฉลี่ยของ Gradient ยกกำลังสอง) มารวมกัน

## 3. การลด Overfitting ด้วย Regularization
- **$\ell_1$ และ $\ell_2$ Regularization**
- **Dropout:** "ปิดใช้งาน" สุ่มเซลล์ประสาทบางตัว (เช่น 50%) ในทุกๆ รอบการ Train ทำให้เซลล์ประสาทที่เหลือต้องพึ่งพาตัวเองมากขึ้น ไม่สามารถพึ่งพาแค่ฟีเจอร์ใดฟีเจอร์หนึ่งได้ ช่วยลด Overfitting ได้ดีมาก
- **Max-Norm Regularization:** จำกัดขนาดของ Weight
