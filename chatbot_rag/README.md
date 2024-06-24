This is demo project for chatbot utilizing RAG as main knowledge source.

## Specification

Python version 3.11.2
OS: Debian GNU/Linux 12 (bookworm)

## Installation

1. Get OpenAI key - [link](https://platform.openai.com/docs/overview)

2. Install docker, docker-compose - [link](https://docs.docker.com/compose/)

3. Install nvidia-container-toolkit, which is required for training Cross Encoder model using GPU in docker container - [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.15.0/install-guide.html)

4. Copy .env file from .sample_env and paste your OpenAI key.

5. After all this steps you should be able to run `docker compose up` to run whole project. 
```