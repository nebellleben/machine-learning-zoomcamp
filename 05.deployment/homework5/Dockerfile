FROM agrigorev/zoomcamp-model:2025
# add your stuff here:
# Install all the dependencies from pyproject.toml
# Copy your FastAPI script
# Run it with uvicorn

WORKDIR /app

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "pipeline_v1.bin", "pyproject.toml", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
#uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
ENTRYPOINT ["uvicorn", "predict:app", "--host 0.0.0.0", "--port 9696", "--reload"]