#!/bin/bash

# Function to check if a command is available
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check if Docker is installed, install if not
if ! command_exists docker; then
    read -p "Docker is not installed. Do you want to install Docker? (y/N): " choice
    case "$choice" in
        y|Y ) 
            # Install Docker
            echo "Installing Docker..."
            # Add Docker's official GPG key:
            sudo apt-get update
            sudo apt-get install ca-certificates curl
            sudo install -m 0755 -d /etc/apt/keyrings
            sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
            sudo chmod a+r /etc/apt/keyrings/docker.asc

            # Add the repository to Apt sources:
            echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
            $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt-get update
            sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

            echo "Adding docker to the user group..."
            sudo groupadd docker
            sudo usermod -aG docker $USER
            newgrp docker
            echo "Docker installed successfully."
            ;;
        * )
            echo "Docker installation skipped."
            ;;
    esac
fi


# 2. Create minio network and run minio 
docker network create minio-network

docker run -d --rm --name minio \
    --network minio-network \
    --env-file minio.cfg \
    -p 9002:9000 \
    -p 9001:9001 \
    -v minio_data:/data \
    minio/minio server /data --console-address ":9001"
