FROM python:3.7-slim

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY model/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model

WORKDIR /app/model/best_model

EXPOSE 5000

CMD ["python3", "server.py"]