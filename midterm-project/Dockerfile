FROM python:3.12.1

RUN pip install uv

WORKDIR /app

COPY ".python-version", "pyprojecdt.toml", "uv.lock", "./"
RUN uv sync --locked

COPY "*.py" "score_predict_model_lr.pkl" "StudentsPerformance.csv" "./"

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]