#!/bin/bash

# Function to check if a command is available
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Create log files
echo "Creating log files..."
mkdir -p logs  # Create logs directory if it doesn't exist
touch logs/vector_db_interface.log
touch logs/chatbot.log
echo "Log files created."

# 2. Check if Docker is installed, install if not
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

#3. Check if Docker-compose is installed, install if not
if ! (command_exists docker-compose || command_exists docker compose); then
    read -p "Docker-compose is not installed. Do you want to install Docker-compose? (y/N): " choice
    case "$choice" in
        y|Y ) 
            # Install Docker
            echo "Installing docker-compose..."
            # Add Docker's official GPG key:
            sudo apt-get install docker-compose-plugin
            ;;
        * )
            echo "docker-compose installation skipped."
            ;;
    esac
fi


# 4. Check if Nvidia-container-toolkit is installed, install if not
if ! command_exists nvidia-container-toolkit; then
    read -p "Nvidia-container-toolkit is not installed. Do you want to install it? (y/N): " choice
    case "$choice" in
        y|Y )
            # Install Nvidia-container-toolkit
            echo "Installing Nvidia-container-toolkit..."
            # Add Nvidia-container-toolkit installation steps here
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            echo "Nvidia-container-toolkit installed successfully."
            echo "Restarting Docker service..."
            #Need to restart because of this https://forums.developer.nvidia.com/t/could-not-select-device-driver-with-capabilities-gpu/80200/3
            sudo systemctl restart docker
            ;;
        * )
            echo "Nvidia-container-toolkit installation skipped."
            ;;
    esac
fi

# 5. Generate .env file and set OpenAI key
#!/bin/bash

# Path to .minio_config in the root directory
MINIO_CONFIG_PATH="../minio.cfg"

# Check if .minio_config exists
if [ ! -f "$MINIO_CONFIG_PATH" ]; then
    echo "Error: minio.cfg not found in the upper directory."
    exit 1
fi

# Load MinIO configuration from .minio_config
export $(grep -v '^#' "$MINIO_CONFIG_PATH" | xargs)

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Generating .env file..."
    cp .sample_env .env
    
    # Prompt for OpenAI key and update .env
    read -p "Please enter your OpenAI key: " openai_key
    sed -i "s/OPENAI_API_KEY=/OPENAI_API_KEY=$openai_key/" .env
    echo "" >> .env
    # Add MinIO configurations to .env
    {
        echo "MINIO_ROOT_USER=$MINIO_ROOT_USER"
        echo "MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD"
        echo "MINIO_DEFAULT_BUCKETS=$MINIO_DEFAULT_BUCKETS"
        echo "MINIO_URL=$MINIO_URL"
    } >> .env

    echo "Generated .env file with OpenAI key and MinIO credentials."
else
    echo ".env file already exists. Skipping generation."
fi

# Proceed with other setup steps
echo "Setup completed successfully."
echo "You can now run 'docker-compose up --build' to start the project."