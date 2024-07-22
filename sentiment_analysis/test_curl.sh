echo "Run training..."
.dQ7-b7&2@wiPpY
.dQ7-b7&2@wiPpY
curl -X POST "http://127.0.0.1:3333/train/" \
-H "Content-Type: application/json" \
-d '{
  "dataset": {
    "text": [
      "I love this product!",
      "This is the worst experience I have ever had.",
      "Absolutely fantastic!",
      "Not good, not bad, just okay.",
      "Terrible, would not recommend.",
      "Great service and friendly staff."
    ],
    "label": [2, 0, 2, 1, 0, 2]
  },
  "model_name": "custom_sentiment_model"
}'

echo "Run inference with trained model..."

curl -X POST "http://127.0.0.1:3333/predict/" \
-H "Content-Type: application/json" \
-d '{"text": "I really enjoyed the service!", "model_name": "custom_sentiment_model"}'

