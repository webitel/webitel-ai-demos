# Configuration
API_URL="http://localhost:1444/api/projects"
LABELING_PROJECTS_DIR="labeling_projects"
USERNAME="webitel.dev@gmail.com"
PASSWORD="some_password"

# Run Label Studio Docker container
docker run -d --rm -p 1444:8080 \
    --network minio-network \
    -e LABEL_STUDIO_USERNAME="$USERNAME" \
    -e LABEL_STUDIO_PASSWORD="$PASSWORD" \
    -e LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true \
    heartexlabs/label-studio:latest \
    label-studio start --username "$USERNAME" --password "$PASSWORD"