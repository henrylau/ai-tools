FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .
# MiVOLO ships a bare setup.py whose first lines import pkg_resources, which
# setuptools removed in 82. Pinning setuptools before requirements.txt does not
# help — resolving requirements.txt pulls a newer setuptools over the top of it.
# Constrain MiVOLO's *build* environment instead, so it builds against a
# setuptools that still has pkg_resources while the runtime keeps the current one.
RUN echo "setuptools<82" > /tmp/build-constraints.txt \
    && uv pip install --system --no-cache wheel "cython<3" \
    && uv pip install --system --no-cache -r requirements.txt \
    && uv pip install --system --no-cache --no-deps \
        --build-constraints /tmp/build-constraints.txt \
        git+https://github.com/WildChlamydia/MiVOLO.git \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app.py test.html ./

# Pre-download model files at build time (download only, no loading into memory)
RUN python -c "\
from huggingface_hub import snapshot_download, hf_hub_download; \
snapshot_download('iitolstykh/mivolo_v2'); \
hf_hub_download('iitolstykh/YOLO-Face-Person-Detector', 'yolov8x_person_face.pt'); \
print('Models downloaded successfully')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]