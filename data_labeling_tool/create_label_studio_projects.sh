#!/bin/bash

# Configuration
API_URL="https://localhost:1444/api/projects"
LABELING_PROJECTS_DIR="labeling_projects"
# 9f12d672d69729ded1b3ff09189b1710e32a7b45
# Prompt for API token
read -p "Enter your Label Studio API token: " TOKEN

# Function to create a project using curl
create_project() {
    local label_config="$1"
    local project_title="$2"
    # Escape special characters in the label_config
    escaped_label_config=$(printf '%s' "$label_config" | sed 's/"/\\"/g' | jq -Rs .)
    echo "Sending payload: {\"label_config\": \"$label_config\"}"
    curl -X POST http://localhost:1444/api/projects/ \
        -H "Authorization: Token  $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"label_config\": \"$label_config\", \"title\": \"$project_title\"}"

    echo $label_config
}

# Iterate through HTML files in the labeling_projects directory
for file in "$LABELING_PROJECTS_DIR"/*.html; do
    if [[ -f "$file" ]]; then
        label_config=$(<"$file")
        
        # Extract project title from the file name up to the second underscore
        filename=$(basename "$file" .html)
        project_title=$(echo "$filename" | cut -d'_' -f1-2)
        
        echo "Creating project with config from $(basename "$file")..."
        create_project "$label_config" "$project_title"
    fi
done
