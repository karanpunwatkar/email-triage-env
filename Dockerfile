# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pydantic asyncio openai

# Expose port if needed (optional)
EXPOSE 8080

# Set environment variables (can be overridden in deployment)
ENV HF_TOKEN=""
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV API_BASE_URL="https://api.openai.com/v1"

# Command to run the inference
CMD ["python", "inference.py"]