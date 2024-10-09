# Use the official Python image as a base
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

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api.py ./

# Expose the API port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
