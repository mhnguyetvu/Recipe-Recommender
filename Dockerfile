FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install prometheus client
RUN pip install prometheus-client

# Copy application code
COPY . .

# Expose port
EXPOSE 2222

# Run application
CMD ["python", "main.py", "--mode", "serve"]
