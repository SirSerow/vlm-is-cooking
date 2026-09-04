FROM python:3.12.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/kitchen
COPY environments/sam3/ environments/sam3/
RUN bash environments/sam3/bootstrap.sh
ENV PATH="/workspace/venvs/sam3/bin:${PATH}" \
    HF_HOME="/workspace/cache/huggingface" \
    PYTHONUNBUFFERED=1
COPY config/ config/
COPY scripts/ scripts/
CMD ["bash"]
