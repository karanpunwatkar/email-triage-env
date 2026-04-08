# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Hugging Face Spaces usually use 7860)
EXPOSE 7860

# Set environment variables (can be overridden in deployment)
ENV HF_TOKEN=""
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV API_BASE_URL="https://api.openai.com/v1"

# Run the app. app.py is in my_env dir, so we use my_env.app:app
CMD ["uvicorn", "my_env.app:app", "--host", "0.0.0.0", "--port", "7860"]