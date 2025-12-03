FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime

COPY hair_classifier_empty.onnx.data .
COPY hair_classifier_empty.onnx .

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]