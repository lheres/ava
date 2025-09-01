# Use the official Python 3.10 slim image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python libraries from the requirements file
# This is a cleaner way to manage dependencies.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache the AI model during the build process.
# This is a crucial optimization for Hugging Face Spaces.
RUN python -c "from transformers import TFGPT2LMHeadModel, GPT2Tokenizer; TFGPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')"

# Copy the rest of the application files
COPY turing_ai.py .
COPY README.md .

# Expose the port that Gradio will run on.
# This is required by Hugging Face Spaces.
EXPOSE 7860

# Set the default command to run the Gradio web app.
CMD ["python", "turing_ai.py"]
