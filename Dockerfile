# syntax=docker/dockerfile:1.5

# 1) Use official PyTorch image with CUDA for GPU support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 2) Install uv
RUN pip install uv

# 3) Set working directory
WORKDIR /app

# 4) Copy pyproject.toml + lock first (better caching)
COPY pyproject.toml uv.lock* ./

# 5) Install dependencies using uv
RUN uv sync

# 6) Copy the rest of your application code
COPY . .

# 7) Expose FastAPI port
EXPOSE 8000

# 8) Command to run FastAPI app with uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["/app/.venv/bin/fastapi", "run", "main.py"]
