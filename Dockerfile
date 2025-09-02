# Use the official Python 3.10 slim image as a base
# Start from the official Python 3.10 slim image.
# This version is compatible with the requested libraries.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set an environment variable for the transformers cache to a writable directory
ENV HF_HOME="/app/cache"

# --- Caching Layer ---
# 1. Copy requirements and install dependencies. This layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy and run the model download script. This layer is cached unless the script changes.
# This downloads the large model files so they don't need to be downloaded on every startup.
COPY download_model.py .
RUN python download_model.py
# --- End Caching Layer ---

# Copy the main application file into the container
COPY turing_ai.py .

# Expose the port that Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "turing_ai.py"]
