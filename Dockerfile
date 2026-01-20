# Forge - Automated Feature Engineering Platform
# Docker image with Jupyter notebook environment

FROM python:3.11-slim

LABEL maintainer="Forge Team <team@forge.dev>"
LABEL description="Forge - Automated Feature Engineering Platform for Machine Learning"
LABEL org.opencontainers.image.source="https://github.com/forge-features/forge"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 1000 forge
USER forge
WORKDIR /home/forge

# Set up PATH for user-installed packages
ENV PATH="/home/forge/.local/bin:${PATH}"

# Copy requirements first for better caching
COPY --chown=forge:forge pyproject.toml README.md /home/forge/forge/
COPY --chown=forge:forge src /home/forge/forge/src

# Install forge with all extras and Jupyter
RUN pip install --user /home/forge/forge[all] jupyter jupyterlab

# Copy notebooks for interactive examples
COPY --chown=forge:forge notebooks /home/forge/notebooks

# Create working directory for user data
RUN mkdir -p /home/forge/work
WORKDIR /home/forge/work

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
