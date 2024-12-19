FROM python:3.10.12-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH} \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

# Create user
RUN useradd -m -u 1000 user

# Install dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Switch to non-root user
USER user
WORKDIR ${HOME}/app

# Install Python packages
COPY --chown=1000 requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user Cython packaging ninja && \
    pip install --no-cache-dir --user youtokentome --no-build-isolation && \
    pip install --no-cache-dir --user -r /tmp/requirements.txt && \
    pip install --no-cache-dir --user fastapi uvicorn pydantic nemo_toolkit && \
    rm -rf /tmp/requirements.txt ~/.cache/pip/*

# Copy application code
COPY --chown=1000 . ${HOME}/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]