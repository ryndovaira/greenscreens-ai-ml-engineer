FROM python:3.12-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    H2O_PORT=54321 \
    H2O_WEB_PORT=54321 \
    H2O_DATA_DIR=/app/dataset \
    H2O_LOG_DIR=/app/logs

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y openjdk-17-jre curl && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy only requirements for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose H2O web UI and API port
EXPOSE 54321

# Set default command
CMD ["python", "train_h2o.py"]
