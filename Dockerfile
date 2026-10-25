FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/opt/u2net

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the rembg model into the image (rembg checks its hash) so nothing is downloaded at runtime.
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')" && ls -l /opt/u2net

COPY app ./app
COPY web ./web
COPY run.py ./run.py
COPY README.md ./README.md

EXPOSE 8000

CMD ["python", "run.py"]
