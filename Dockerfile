# Simple CPU container for running the NMT pipeline.

FROM python:3.10-slim

# Keep Python output unbuffered and avoid .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first for layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project into the container.
COPY . /app

# Default to the pipeline folder for commands.
WORKDIR /app/nmt-pipeline

CMD ["bash"]

