FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

#Install pip
RUN apt-get update && apt-get install -y \
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

RUN pip install --no-cache-dir fastapi uvicorn
RUN pip install --no-cache-dir python-multipart
RUN pip install --no-cache-dir huggingface_hub pyctcdecode 
RUN apt-get update && apt-get install -y build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
RUN pip install https://github.com/kpu/kenlm/archive/master.zip


COPY ./ ./


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000",  "--workers","1"]