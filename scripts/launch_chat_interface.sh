docker run -d \
  --name chat_interface \
  --network webitel-ai-demos_common-network \
  --env-file .env \
  -v ./prompts:/app/prompts:ro \
  -v ./logs/chatbot.log:/app/chatbot.log \
  -p 50055-50056:50055 \
  --restart always \
  chat_interface
