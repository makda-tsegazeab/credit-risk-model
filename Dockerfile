# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY ./src ./src
COPY ./models ./models

# Expose port
EXPOSE 8000

# Command to run FastAPI with reload
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
