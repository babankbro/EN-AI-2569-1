# Ensemble Learning and Random Forests

จากบทที่ 7 Ensemble Learning and Random Forests

## คอนเซปต์หลัก
- **ภูมิปัญญาฝูงชน (Wisdom of the crowd):** นำคำทำนายจากหลายๆ โมเดลมารวมกัน มักจะได้ผลลัพธ์ที่ดีกว่าโมเดลเดี่ยวๆ ที่ดีที่สุด
- โมเดลกลุ่มนี้เรียกว่า **Ensemble methods**

## เทคนิคต่างๆ

### 1. Voting Classifiers
- เอารุ่นต่างๆ มาทำนาย (เช่น Logistic + SVM + Random Forest)
- **Hard Voting:** เอาผลคำทำนายแบบเด็ดขาด (Class) มาโหวตกัน ใครได้เสียงข้างมากชนะ
- **Soft Voting:** เอาค่าความน่าจะเป็น (Probability) มาเฉลี่ยกัน แล้วเลือกคลาสที่เฉลี่ยสูงสุด มักได้ผลดีกว่า Hard Voting

### 2. Bagging and Pasting
- ฝึกสอนโมเดลประเภทเดียวกัน (เช่น Decision Tree อย่างเดียว) หลายๆ ตัว แต่ให้แต่ละตัว **เรียนรู้จากข้อมูลคนละชุดที่ถูกสุ่มขึ้นมา**
- **Bagging (Bootstrap aggregating):** สุ่มข้อมูลโดยมี "การใส่กลับ" (with replacement) - *นิยมใช้มากที่สุด*
- **Pasting:** สุ่มข้อมูลแบบ "ไม่ใส่กลับ" (without replacement)

### 3. Random Forests
- คือการทำ Bagging กับโมเดล Decision Trees หลายๆ ตัว (เช่น 500 ต้น)
- เป็นอัลกอริทึมที่ทรงพลังมาก และสามารถบอกความสำคัญของฟีเจอร์ (Feature Importance) ได้

### 4. Boosting
- สร้างโมเดลต่อๆ กันไปเรื่อยๆ โดยโมเดลตัวถัดไปจะพยายาม **"แก้ไขข้อผิดพลาด"** ของโมเดลตัวก่อนหน้า
- **AdaBoost:** ให้ความสำคัญ (Weight) กับข้อมูลตัวที่ทายผิดมากขึ้น
- **Gradient Boosting:** สร้างโมเดลตัวใหม่ให้ไปทำนายส่วนต่าง/ความผิดพลาด (Residual errors) ของตัวเดิม

## ตัวอย่างโค้ด Random Forest (Scikit-Learn)
```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

# ดูความสำคัญของแต่ละ Feature
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```
