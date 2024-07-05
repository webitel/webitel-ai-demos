#!/bin/sh

# Log file location
LOG_FILE=".pre-commit-setup.log"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

log "Starting pre-commit hook setup..."

# Check if pre-commit is installed
if ! command -v pre-commit > /dev/null 2>&1; then
    log "pre-commit is not installed. Installing pre-commit..."
    if pip install pre-commit >> $LOG_FILE 2>&1; then
        log "pre-commit installed successfully."
    else
        log "Failed to install pre-commit. Please install it manually."
        exit 1
    fi
else
    log "pre-commit is already installed."
fi

# Install pre-commit hooks
log "Installing pre-commit hooks..."
if pre-commit install >> $LOG_FILE 2>&1; then
    log "pre-commit hooks installed successfully."
else
    log "Failed to install pre-commit hooks. Check the log for details."
    exit 1
fi

# Create post-checkout hook
log "Setting up post-checkout hook..."
POST_CHECKOUT_HOOK=".git/hooks/post-checkout"
cat << 'EOF' > $POST_CHECKOUT_HOOK
#!/bin/sh
sh setup-hooks.sh
EOF
chmod +x $POST_CHECKOUT_HOOK
log "post-checkout hook set up successfully."

# Create post-merge hook
log "Setting up post-merge hook..."
POST_MERGE_HOOK=".git/hooks/post-merge"
cat << 'EOF' > $POST_MERGE_HOOK
#!/bin/sh
sh setup-hooks.sh
EOF
chmod +x $POST_MERGE_HOOK
log "post-merge hook set up successfully."

log "Pre-commit hook setup completed."

exit 0

