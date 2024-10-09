docker run -d \
  --name ner-service \
  --network webitel-ai-demos_common-network \
  --env-file .env \
  --restart always \
  --gpus all \
  ner_service
