FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip "setuptools<70" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

COPY app.py .

# Pre-download model files at build time (download only, no loading into memory)
RUN python -c "\
from huggingface_hub import snapshot_download, hf_hub_download; \
snapshot_download('iitolstykh/mivolo_v2'); \
hf_hub_download('iitolstykh/YOLO-Face-Person-Detector', 'yolov8x_person_face.pt'); \
print('Models downloaded successfully')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
