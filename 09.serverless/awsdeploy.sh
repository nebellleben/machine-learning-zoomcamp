aws ecr create-repository --repository-name "hair-classifier" --region "us-east-1"

ECR_URL="808055627316.dkr.ecr.us-east-1.amazonaws.com"

aws ecr get-login-password \
  --region "us-east-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

REMOTE_IMAGE_TAG="${ECR_URL}/hair-classifier:v1"

docker build -t hair-classifier .
docker tag hair-classifier ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

# AI generated: IAM creation

aws iam create-role \
  --role-name lambda-ecr-role \
  --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
          "Effect": "Allow",
          "Principal": {"Service": "lambda.amazonaws.com"},
          "Action": "sts:AssumeRole"
      }]
  }'

aws iam attach-role-policy \
  --role-name lambda-ecr-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# AI generated: Lambda function creation