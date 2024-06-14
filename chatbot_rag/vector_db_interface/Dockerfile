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

RUN pip install langchain_milvus langchain_community
# Copy the application code

RUN pip install weaviate-client
COPY . .

#Generate grcp files
RUN python -m grpc_tools.protoc -I .   --python_out=. --grpc_python_out=.   vector_db.proto

# Set the entrypoint command
CMD ["python", "vector_db_interface_grpc_server.py"]