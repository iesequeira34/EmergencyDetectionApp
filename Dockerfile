FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync

COPY . .

EXPOSE 8000

CMD ["/app/.venv/bin/fastapi", "run", "main.py"]
