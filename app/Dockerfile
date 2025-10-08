# Use Python 3.11.5 Bookworm variant
FROM python:3.11.5-bookworm

# Set Python environment variables to prevent writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install ffmpeg, a system dependency for openai-whisper.
# We run update, install, and cleanup in a single RUN command to reduce image layers.
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching.
COPY ./app/requirements.txt .

# Upgrade pip and install dependencies with a longer timeout
# Using --no-cache-dir keeps the image size smaller.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Now copy the rest of the application code into the working directory
COPY ./app .

# Create a non-root user `appuser` and assign ownership of the /app directory
# This must be done as root before switching the user.
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

# Switch to the non-root `appuser` for improved security
USER appuser

# Expose port 80 for the application.
EXPOSE 80

# Set uvicorn as the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
