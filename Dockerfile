# Use the official Python 3.10 slim image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install the requested Python libraries plus 'transformers' for the AI model
# We use --no-cache-dir to reduce the image size
RUN pip install --no-cache-dir pandas tensorflow numpy transformers

# Download and cache the AI model during the build process.
# This saves time when the container starts. We're using GPT-2, a powerful
# and well-known model from OpenAI.
RUN python -c "from transformers import TFGPT2LMHeadModel, GPT2Tokenizer; TFGPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')"

# Copy the AI script and README into the container
COPY turing_ai.py .
COPY README.md .

# Set the default command to run when the container starts.
# This will automatically launch our AI.
CMD ["python", "-u", "turing_ai.py"]

