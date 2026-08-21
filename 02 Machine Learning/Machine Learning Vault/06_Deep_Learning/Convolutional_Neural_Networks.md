# Convolutional Neural Networks (CNNs)

จากบทที่ 14 Deep Computer Vision Using Convolutional Neural Networks

## ทำไมต้อง CNN?
- ทฤษฎีมาจาก **Visual Cortex** ของสัตว์ (Hubel & Wiesel) 
- ถ้านำภาพใหญ่ๆ ไปใช้กับ Neural Network ปกติ (Dense layer) จำนวน Parameters จะมหาศาลมาก (เช่น ภาพ $100 \times 100$ เข้า $1000$ โหนด = $10$ ล้านพารามิเตอร์ในชั้นแรก)
- CNN แก้ปัญหานี้ด้วย **Partially connected layers** และ **Weight sharing** (ใช้ฟิลเตอร์ตัวเดียวกันกวาดไปทั่วภาพ)

## โครงสร้างพื้นฐาน
1. **Convolutional Layer:**
   - ใช้ **Filter** (หรือ Kernel) กวาด (Slide) ไปทั่วภาพเพื่อตรวจจับ Feature เช่น เส้นแนวตั้ง แนวนอน 
   - เกิดเป็น **Feature Map** ใหม่
   - สามารถซ้อน (Stack) หลายๆ Feature map ได้เพื่อตรวจจับหลาย Feature (เช่น มี 32 ฟิลเตอร์ ก็จะได้ 32 Feature maps)
   - พารามิเตอร์สำคัญ: `filters`, `kernel_size`, `strides`, `padding` (VALID/SAME)

2. **Pooling Layer:**
   - ย่อส่วนภาพ (Subsample) ลง เพื่อลดขนาดข้อมูล ลดพารามิเตอร์ และป้องกัน Overfitting
   - ทำให้เกิด Translation Invariance (เลื่อนรูปไปซ้ายขวานิดหน่อยก็ยังทายถูก)
   - ชนิดที่พบบ่อย: **Max Pooling** (เอาค่ามากสุดในบริเวณนั้นมาใช้)

## สถาปัตยกรรมระดับโลก (CNN Architectures)
- **LeNet-5 (1998):** ต้นตำรับสำหรับอ่านตัวเลข
- **AlexNet (2012):** คล้าย LeNet-5 แต่ใหญ่กว่า ลึกกว่า ใช้ ReLU และ Dropout ทำให้ Deep Learning กลับมาบูม
- **GoogLeNet (2014):** ใช้ **Inception Modules** (ใช้ Filter หลายขนาดพร้อมๆ กันแล้วเอามาต่อกัน)
- **ResNet (2015):** ชนะด้วยโมเดลที่ลึกมาก (152 ชั้น) โดยใช้ **Skip Connections** (เอาค่า Input มาบวกเข้ากับ Output ตรงๆ เลย) ช่วยแก้ปัญหา Vanishing Gradient
- อื่นๆ: VGGNet, Xception, SENet

## ตัวอย่างโค้ด Keras
```python
from tensorflow import keras

model = keras.models.Sequential([
    # Convolutional Layer + Pooling
    keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    
    # Fully Connected Network
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
```
