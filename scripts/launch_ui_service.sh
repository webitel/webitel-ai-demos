docker run -d \
  --name ui-service \
  --network webitel-ai-demos_common-network \
  --env-file .env \
  -v ./prompts:/app/prompts:ro \
  -p 7860:7860 \
  --restart always \
  ui_service
