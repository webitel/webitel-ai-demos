FROM python:slim

#Updates
RUN apt-get update -y
RUN apt-get upgrade -y

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install minio

RUN pip install stopit

#Generate grcp files
RUN ./generate_grpc.sh

# Set the entrypoint command
CMD ["python", "chat_interface_grpc_server.py"]