docker run -d --rm --name stt-service -p 4444:5000 --gpus all  --network webitel-ai-demos_common-network --env-file .env stt_service