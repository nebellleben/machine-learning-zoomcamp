FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime

COPY hair_classifier_empty.onnx.data .
COPY hair_classifier_empty.onnx .

CMD ["homework9.lambda_handler"]