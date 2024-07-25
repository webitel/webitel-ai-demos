# Webitel AI Projects Repository

Welcome to the Webitel AI Projects repository. This repository contains all AI-related projects developed by Webitel.
# Setup Instructions

To ensure code quality and consistency, we use pre-commit hooks. Please follow the steps below to set up the hooks for this project:
## Setting Up Pre-commit Hooks

    Clone the repository:

```
git clone <repository-url>
cd <repository-directory>
```
Run the setup script:
For Unix-like systems:

```
./setup-hooks.sh
```
    For Windows systems, you may need to use Git Bash or WSL, or run a batch script (to be created separately if needed).

The setup script will install the necessary pre-commit hooks and set up Git hooks to ensure they are installed on checkout and merge operations. This helps ensure successful commits by enforcing code standards.

## Starting services 

A lot of our services utilize minio, so you need to start it.
You can also observe minio config - minio.cfg 

Run minio (via docker):

```
./start_minio.sh
```

Here is the full list of services that utilize minio :

- Rag Chatbot
- Data Labeling Tool

