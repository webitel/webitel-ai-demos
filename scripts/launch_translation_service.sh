docker run -d \
  --name translation-service \
  --network webitel-ai-demos_common-network \
  --env-file .env \
  -p 8251:8000 \
  --restart always \
  -v translation_model_data:/root/.cache/huggingface/transformers \
  --gpus all \
  translation_service
