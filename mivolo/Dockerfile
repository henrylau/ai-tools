FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps git+https://github.com/WildChlamydia/MiVOLO.git

COPY app.py .

# Pre-download models at build time so startup is fast
RUN python -c "\
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification; \
AutoConfig.from_pretrained('iitolstykh/mivolo_v2', trust_remote_code=True); \
AutoImageProcessor.from_pretrained('iitolstykh/mivolo_v2', trust_remote_code=True); \
AutoModelForImageClassification.from_pretrained('iitolstykh/mivolo_v2', trust_remote_code=True); \
import huggingface_hub; \
huggingface_hub.hf_hub_download('iitolstykh/YOLO-Face-Person-Detector', 'yolov8x_person_face.pt'); \
print('Models downloaded successfully')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
