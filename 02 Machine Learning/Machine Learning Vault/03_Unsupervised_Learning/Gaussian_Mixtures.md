# Gaussian Mixture Models (GMM)

จากบทที่ 9 Unsupervised Learning Techniques

## คอนเซปต์หลัก
- **GMM** คือโมเดลที่ตั้งสมมติฐานว่าข้อมูล (Instances) ทั้งหมดถูกสร้างมาจาก "ส่วนผสม" (Mixture) ของการแจกแจงแบบเกาส์ (Gaussian distributions) หลายๆ ตัวที่มีพารามิเตอร์ไม่ทราบค่า
- ในการจัดกลุ่ม แต่ละ Cluster ก็คือแต่ละ Gaussian distribution ซึ่งโดยปกติมักมีรูปร่างเป็นวงรี (Ellipsoid)
- ข้อมูลที่ถูกสร้างจากการแจกแจงอันเดียวกันก็จะรวมเป็น Cluster เดียวกัน
- ดังนั้น GMM สามารถใช้หา Cluster ที่มีรูปร่างวงรี ทั้งขนาด, ความหนาแน่น, และการหมุน (Orientation) ที่แตกต่างกันได้ (ยืดหยุ่นกว่า K-Means มาก)

## การใช้งาน
- นิยมใช้มากสำหรับ **Density Estimation** และการทำ **Anomaly Detection**
- สามารถระบุ Outlier ได้ง่ายๆ โดยดูว่าจุดข้อมูลไหนอยู่ในพื้นที่ที่มี "ความหนาแน่น (Density) ต่ำ" ตามโมเดล
- ฝึกสอน (Train) โดยใช้อัลกอริทึม Expectation-Maximization (EM) ซึ่งมีสองขั้นตอนคล้าย K-Means

**ตัวอย่างโค้ด GMM สำหรับ Anomaly Detection (Scikit-Learn)**
```python
from sklearn.mixture import GaussianMixture
import numpy as np

# ฝึกสอนโมเดล GMM สมมติว่ามี 3 กลุ่ม
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

# คำนวณหา Density Score (ยิ่งมากยิ่งแปลว่าหนาแน่น)
densities = gm.score_samples(X)

# หาเกณฑ์เพื่อตัด Outliers (เช่น ล่างสุด 4%)
density_threshold = np.percentile(densities, 4)

# จุดไหนที่มีค่าน้อยกว่า Threshold จะถือว่าเป็น Anomaly
anomalies = X[densities < density_threshold]
```
