# Introduction to Artificial Neural Networks (ANNs)

จากบทที่ 10 Introduction to Artificial Neural Networks with Keras

## คอนเซปต์หลัก
- **ANNs** ได้แรงบันดาลใจมาจากเซลล์ประสาทในสมองของสัตว์
- สถาปัตยกรรมที่ง่ายที่สุดคือ **Perceptron** ซึ่งประกอบด้วยชั้นของ TLU (Threshold Logic Unit)
- **Multi-Layer Perceptron (MLP)** ประกอบด้วย:
  1. Input layer
  2. Hidden layers (อย่างน้อย 1 ชั้น)
  3. Output layer
- เมื่อ MLP มี Hidden layers หลายชั้น จะเรียกว่า **Deep Neural Network (DNN)**

## Backpropagation Algorithm
- อัลกอริทึมหัวใจสำคัญที่ใช้สอน MLP (ย้อนกลับไปปรับค่าน้ำหนัก)
- **ขั้นตอนคร่าวๆ:**
  1. **Forward pass:** ส่งข้อมูลเข้าโมเดลเพื่อทำนายผล และวัดความผิดพลาด (Error)
  2. **Reverse pass:** วัดว่าจุดเชื่อมต่อแต่ละจุด (Connections/Weights) มีส่วนทำให้เกิด Error เท่าไหร่ โดยใช้กฎลูกโซ่ (Chain rule)
  3. **Gradient Descent step:** ปรับค่าน้ำหนักเพื่อลด Error

## Activation Functions
การส่งข้อมูลข้ามชั้น (Layer) จำเป็นต้องมี "ความไม่เป็นเชิงเส้น" (Non-linearity) ไม่งั้นต่อให้มีร้อยชั้นก็มีค่าเท่ากับชั้นเดียว
- **Sigmoid (Logistic):** ให้ค่า 0 ถึง 1 มักใช้กับ Output layer สำหรับงาน Binary Classification
- **Tanh:** ให้ค่า -1 ถึง 1
- **ReLU (Rectified Linear Unit):** ให้ค่า 0 ถ้าติดลบ นอกนั้นให้ค่าเดิม นิยมใช้ที่สุดสำหรับ Hidden layers เพราะทำงานเร็วและช่วยลดปัญหา Vanishing Gradients
- **Softmax:** มักใช้กับ Output layer สำหรับงาน Multiclass Classification

## ตัวอย่างโค้ด Keras (Sequential API)
```python
from tensorflow import keras

# สร้างโมเดลแบบ Classification (เช่น MNIST)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]), # เปลี่ยนภาพ 2D เป็น 1D Array
    keras.layers.Dense(300, activation="relu"), # Hidden layer 1
    keras.layers.Dense(100, activation="relu"), # Hidden layer 2
    keras.layers.Dense(10, activation="softmax") # Output layer (10 classes)
])

# กำหนด Loss function และ Optimizer
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# ฝึกสอนโมเดล (มี Validation set)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
```
