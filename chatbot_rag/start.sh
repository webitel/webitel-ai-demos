#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Copying proto files to each microservice directory..."

# Define an array of microservice directories
services=("vector_db_interface" "chat_interface" "ui_service" "translation_service") # "chatbot_evaluation")

# Loop through each service and copy the proto files
for service in "${services[@]}"; do
  cp -r protos "$service/"
  echo "Copied proto files to $service/"
done

echo "Starting Docker Compose services..."

# Start Docker Compose services
docker compose up --build

