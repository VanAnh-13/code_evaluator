# Multi-stage build to reduce image size
FROM python:3.12-slim AS builder

# Set environment variables to reduce size and improve performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# Second stage: final image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /build/wheels /wheels

# Install dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy the project files to the container
COPY . /app/

# Add metadata labels
LABEL maintainer="Your Name <your.email@example.com>" \
      version="1.0" \
      description="C++ Code Analyzer using Qwen model"

# Create a non-root user to run the application
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create directories for the web application
RUN mkdir -p /app/web_app/static/css /app/web_app/static/js /app/web_app/templates

# Copy web application files
COPY web_app/app.py /app/web_app/
COPY web_app/static/css/style.css /app/web_app/static/css/
COPY web_app/static/js/main.js /app/web_app/static/js/
COPY web_app/templates/*.html /app/web_app/templates/

# Expose port for web server
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=/app/web_app/app.py
ENV FLASK_ENV=production

# Set entry point to run the web server
ENTRYPOINT ["python", "web_app/app.py"]
