This is demo project for chatbot utilizing RAG as main knowledge source.

## Specification

Python version 3.11.2
OS: Debian GNU/Linux 12 (bookworm)

## Install requirements

```
python -m venv env

source env/bin/activate

pip install -r requirements.txt 
```

## Run app


# You might need to reload nvidia modules
```
sudo rmmod nvidia_uvm
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```


```
python ui.py
```