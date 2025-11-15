FROM python:3.12.1-slim-bookworm

# RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

COPY "train.py" "predict.py" "testrequest.py" "score_predict_model_lr.pkl" "StudentsPerformance.csv" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]