# Use the official Python 3.10 slim image as a base
# Start from the official Python 3.10 slim image.
# This version is compatible with the requested libraries.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- Environment Tuning & Dependencies ---
# 1. Install system-level dependencies:
#    - procps: for `lscpu` to detect physical cores
#    - libjemalloc-dev: a high-performance memory allocator
RUN apt-get update && apt-get install -y procps libjemalloc-dev && rm -rf /var/lib/apt/lists/*

# 2. Set environment variables to use the jemalloc allocator and for the transformers cache
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV HF_HOME="/app/cache"

# --- Caching Layer ---
# 1. Copy requirements and install dependencies. This layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy and run the model download script. This layer is cached unless the script changes.
# This downloads the large model files so they don't need to be downloaded on every startup.
COPY download_model.py .
RUN python download_model.py
# --- End Caching Layer ---

# Copy the main application file into the container
COPY turing_ai.py .
COPY entrypoint.sh .

# 3. Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# --- Execution ---
# Use the entrypoint script to set CPU-specific ENV VARS and then run the app
ENTRYPOINT ["./entrypoint.sh"]

# Expose the port that Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "turing_ai.py"]
