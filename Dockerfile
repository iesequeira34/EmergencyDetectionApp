FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync

COPY . .

EXPOSE 8000

CMD ["/app/.venv/bin/fastapi", "run", "main.py"]
