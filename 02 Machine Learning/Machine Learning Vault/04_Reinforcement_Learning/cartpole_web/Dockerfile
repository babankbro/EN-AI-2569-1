FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PyTorch แบบ CPU อย่างเดียว (ขนาดเล็กกว่ารุ่น CUDA มาก) สำหรับ Atari DQN
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
# ใช้ Q-table ที่ฝึกไว้แล้วใน app/models/ ถ้าไม่มีให้ฝึกใหม่ตอน build (~1 นาที)
RUN [ -f app/models/qtable.npy ] || python -m app.train --episodes 3000

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
