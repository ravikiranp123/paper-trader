FROM python:3.12-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY . .

# Sync dependencies
RUN uv sync

# Expose port (adjust if needed based on your server configuration)
EXPOSE 8000

# Run the server
CMD ["uv", "run", "server.py"]
