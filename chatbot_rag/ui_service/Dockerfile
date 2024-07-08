# Start from the Python slim image
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY reqs.txt /app

#Install pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install any needed packages specified in requirements.txt
RUN pip install -r reqs.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 7860 available to the world outside this container
EXPOSE 7860

RUN protos/generate_grpc.sh

# Run ui when the container launches
CMD ["python", "ui.py"]
