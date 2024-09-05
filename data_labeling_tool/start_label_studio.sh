# Configuration
API_URL="http://localhost:1444/api/projects"
LABELING_PROJECTS_DIR="labeling_projects"
USERNAME="webitel.dev@gmail.com"
PASSWORD="some_password"

# Run Label Studio Docker container
# make sure that host dir is writable https://github.com/HumanSignal/label-studio/issues/3465#issuecomment-1365707616
docker run -d --rm \
    --network host \
    -e LABEL_STUDIO_USERNAME="$USERNAME" \
    -e LABEL_STUDIO_PASSWORD="$PASSWORD" \
    -e LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true \
    -e ENABLE_CSP=false \
    -e LABEL_STUDIO_BASE_DATA_DIR=/label-studio/labeling_dir \
    -v /mnt/md1/labeling_dir:/label-studio/labeling_dir \
    heartexlabs/label-studio:latest \
    label-studio start --username "$USERNAME" --password "$PASSWORD" --port 1444
    


    
    
