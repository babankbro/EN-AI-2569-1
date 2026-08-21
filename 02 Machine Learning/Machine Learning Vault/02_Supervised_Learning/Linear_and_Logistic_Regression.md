# Linear and Logistic Regression

จากบทที่ 4 Training Models

## 1. Linear Regression
- เป็นการหาเส้นตรง (หรือ Hyperplane ในหลายมิติ) ที่เป็นตัวแทนของข้อมูลได้ดีที่สุด
- โมเดลทำนายด้วยสมการ: $\hat{y} = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$
- **Cost Function:** ใช้ Mean Squared Error (MSE) เป็นตัววัดความผิดพลาดของโมเดล เราต้องการให้ค่านี้ต่ำที่สุด

### การสอนโมเดล (Training)
1. **The Normal Equation:** เป็นสมการคณิตศาสตร์ที่หาคำตอบได้โดยตรง (Closed-form solution)
   - เร็วเมื่อฟีเจอร์น้อย แต่ช้ามากเมื่อจำนวนฟีเจอร์เยอะ
2. **Gradient Descent (GD):** การหาจุดต่ำสุดโดยค่อยๆ ปรับค่าพารามิเตอร์สวนทางกับทิศทางความชัน
   - **Batch GD:** ใช้ข้อมูลทั้งหมดในการปรับแต่ละครั้ง (ช้าแต่แม่น)
   - **Stochastic GD (SGD):** สุ่มหยิบทีละตัวมาปรับ (เร็ว แกว่งไปมา)
   - **Mini-batch GD:** สุ่มหยิบมากลุ่มเล็กๆ (ประนีประนอมระหว่างสองแบบ)
   - อัตราการเรียนรู้ (Learning Rate: $\eta$) คือขนาดก้าวเดิน ถ้าน้อยไปจะช้า ถ้ามากไปจะกระโดดข้ามจุดต่ำสุด

### การแก้ปัญหา Overfitting
ใช้ **Regularization** เพื่อสร้างข้อจำกัดไม่ให้เส้นมันซับซ้อนเกินไป:
- **Ridge Regression:** ลากเส้นโดยพยายามให้ค่าน้ำหนัก (Weights) มีขนาดเล็ก (เพิ่ม L2 penalty)
- **Lasso Regression:** คล้าย Ridge แต่มีแนวโน้มตัดฟีเจอร์ที่ไม่จำเป็นทิ้งให้เป็น 0 (L1 penalty)
- **Elastic Net:** ผสมระหว่าง Ridge กับ Lasso

---

## 2. Logistic Regression
- ใช้สำหรับงาน **Classification (การแยกประเภท)** โดยเฉพาะ Binary Classification
- ให้ผลลัพธ์เป็น "ความน่าจะเป็น" (Probability) ว่าข้อมูลนี้จะตกอยู่ในคลาส 1 (Positive class)
- ใช้ฟังก์ชัน **Sigmoid (Logistic function)** แปลงผลจากสมการเชิงเส้นให้อยู่ในช่วง 0 ถึง 1

### ตัวอย่างโค้ด (Scikit-Learn)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris["data"][:, 3:] # เอาแค่ petal width
y = (iris["target"] == 2).astype(np.int) # พยายามทายว่าเป็น Iris-Virginica หรือไม่

log_reg = LogisticRegression()
log_reg.fit(X, y)

# ทำนายผล
y_proba = log_reg.predict_proba([[1.7]])
print(y_proba) # ความน่าจะเป็น
```
