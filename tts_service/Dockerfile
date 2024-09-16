FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

#Install pip
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*


# Make python3 the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/robinhad/ukrainian-tts.git

COPY ./ ./


CMD ["python","main.py"]