docker run -d --rm --name voice-bot-service -p 2323:8080 --network webitel-ai-demos_common-network --env-file .env voice_bot_service