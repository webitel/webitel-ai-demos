FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

#Install pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Make python3 the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

RUN chmod +x entrypoint.sh

CMD ["bash","entrypoint.sh"]