# Support Vector Machines (SVM)

จากบทที่ 5 Support Vector Machines

## คอนเซปต์หลัก
- **Large Margin Classification:** พยายามสร้างเส้นแบ่ง (Decision Boundary) หรือ "ถนน" ให้กว้างที่สุด เพื่อแบ่งแยกระหว่างคลาส
- ข้อมูลที่อยู่ริมถนนเหล่านี้เรียกว่า **Support Vectors**
- SVM มีความอ่อนไหวต่อสเกลของข้อมูลมาก (ต้องทำ Feature Scaling เสมอ เช่นใช้ StandardScaler)

## ประเภทของ Margin
1. **Hard Margin:** ห้ามมีข้อมูลหลุดเข้ามาในถนนเลย (ใช้ได้เฉพาะกับข้อมูลที่แยกขาดกันได้ 100% และไวต่อ Outlier)
2. **Soft Margin:** ยอมให้มีข้อมูลหลุดเข้ามาได้บ้าง เพื่อให้เส้นแบ่งมีความยืดหยุ่นขึ้น
   - ใน Scikit-Learn ควบคุมความเข้มงวดนี้ด้วยพารามิเตอร์ `C`
   - ค่า `C` น้อย: ถนนกว้างขึ้น (ยอมผิดพลาดได้มากขึ้น - ป้องกัน Overfitting)
   - ค่า `C` มาก: ถนนแคบลง (พยายามทำให้ถูกทุกตัว)

## Nonlinear SVM Classification
ถ้าข้อมูลไม่ได้เป็นเส้นตรงล่ะ?
1. **เพิ่ม Polynomial Features:** เติมตัวแปรยกกำลังเข้าไป ทำให้ข้อมูลสามารถถูกแยกได้ด้วยเส้นตรงในมิติที่สูงขึ้น
2. **The Kernel Trick:** เป็นเวทมนตร์ทางคณิตศาสตร์ (เช่น Polynomial Kernel, Gaussian RBF Kernel) ที่ทำให้ได้ผลลัพธ์เหมือนกับเพิ่มฟีเจอร์ไปจริงๆ โดยไม่ต้องเพิ่มจริง ทำให้ไม่เปลืองทรัพยากรเครื่อง

### ตัวอย่างโค้ด SVM เชิงเส้น (Scikit-Learn)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X, y)
```
