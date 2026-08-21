# Decision Trees

จากบทที่ 6 Decision Trees

## คอนเซปต์หลัก
- ตัดสินใจโดยการตั้งคำถามที่เป็น Yes/No ลงมาเรื่อยๆ จนกว่าจะเจอคำตอบ (Root node -> child node -> leaf node)
- เป็นโมเดลประเภท **White Box Model** เพราะมนุษย์สามารถเข้าใจและอธิบายเหตุผลได้ง่าย (ต่างจาก Neural Network ที่เป็น Black Box)
- เป็นโมเดลพื้นฐานที่ใช้สร้าง Random Forests

## วิธีที่ใช้สร้างต้นไม้ (CART Algorithm)
- Scikit-Learn ใช้ CART (Classification And Regression Tree)
- อัลกอริทึมจะค้นหา "ฟีเจอร์" และ "ค่าเงื่อนไข" ที่ใช้แบ่ง (Split) ข้อมูลออกเป็น 2 กลุ่ม โดยให้ผลลัพธ์ออกมา "บริสุทธิ์" (Pure) ที่สุด
- ความบริสุทธิ์วัดจาก **Gini Impurity** หรือ **Entropy** (ค่ายิ่งต่ำ ยิ่งบริสุทธิ์)

## การป้องกัน Overfitting (Regularization)
Decision Trees เสี่ยงต่อการ Overfit มากถ้ายอมให้มันสร้างกิ่งไปเรื่อยๆอย่างไร้ขอบเขต
- ควบคุมโดยใช้ Hyperparameters เช่น:
  - `max_depth`: ความลึกสูงสุดของต้นไม้
  - `min_samples_split`: จำนวนตัวอย่างขั้นต่ำก่อนที่จะยอมให้สร้างกิ่งเพิ่ม
  - `min_samples_leaf`: จำนวนตัวอย่างขั้นต่ำที่ต้องมีใน leaf node

## ตัวอย่างโค้ด (Scikit-Learn)
```python
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```
