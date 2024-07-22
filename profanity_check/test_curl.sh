echo "Run with default confidence threshold\n"

curl -X POST http://127.0.0.1:3333/search -H "Content-Type: application/json" -d '{
    "words": ["привт", "свт"],
    "text": "Привіт, усі! Ласкаво просимо до світу FastAPI."
}'

echo "\nRun with 90% confidence\n"

curl -X POST http://127.0.0.1:3333/search -H "Content-Type: application/json" -d '{
    "words": ["привт", "свт"],
    "text": "Привіт, усі! Ласкаво просимо до світу FastAPI.",
    "confidence": 90
}'

