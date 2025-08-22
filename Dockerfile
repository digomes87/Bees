# Multi-stage Dockerfile for Bee Migration Analysis Project
# This Dockerfile creates a production-ready container with all dependencies

# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal28 \
    libproj19 \
    libgeos-c1v5 \
    libspatialindex6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r beeuser && useradd -r -g beeuser beeuser

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=beeuser:beeuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/images /app/results /app/logs && \
    chown -R beeuser:beeuser /app

# Switch to non-root user
USER beeuser

# Install the package in development mode
RUN pip install -e .

# Expose port for potential web services
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "-m", "bee_migration.cli", "--help"]

# Development stage
FROM production as development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER beeuser

# Install pre-commit hooks
RUN pre-commit install || true

# Default command for development
CMD ["bash"]

# Jupyter stage
FROM development as jupyter

# Expose Jupyter port
EXPOSE 8888

# Install Jupyter extensions
RUN pip install jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install --user

# Create Jupyter config
RUN jupyter notebook --generate-config

# Set Jupyter configuration
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Default command for Jupyter
CMD ["jupyter", "notebook", "--notebook-dir=/app/notebooks"]

# API stage
FROM production as api

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart

# Expose API port
EXPOSE 8000

# Default command for API
CMD ["uvicorn", "bee_migration.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard stage
FROM production as dashboard

# Install dashboard dependencies
RUN pip install streamlit dash plotly

# Expose dashboard port
EXPOSE 8501

# Default command for dashboard
CMD ["streamlit", "run", "bee_migration/dashboard.py", "--server.address", "0.0.0.0"]

# Testing stage
FROM development as testing

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]

# Labels for metadata
LABEL maintainer="Diego Gomes <diego@example.com>" \
      version="1.0.0" \
      description="Bee Migration Analysis: Climate Impact Assessment" \
      org.opencontainers.image.title="Bee Migration Analysis" \
      org.opencontainers.image.description="Climate Impact Assessment on Bee Migration Patterns" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Diego Gomes" \
      org.opencontainers.image.url="https://github.com/diegogo/bee-migration-analysis" \
      org.opencontainers.image.source="https://github.com/diegogo/bee-migration-analysis" \
      org.opencontainers.image.licenses="MIT"