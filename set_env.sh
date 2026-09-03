#!/bin/bash
PROJECT_FILE="$HOME/project_id.txt"

echo "--- Configuring Agent Evaluation Environment Variables ---"

# 1. Resolve Project ID
if [ -s "$PROJECT_FILE" ]; then
    export PROJECT_ID=$(cat "$PROJECT_FILE" | tr -d '[:space:]')
elif [ -n "$DEVSHELL_PROJECT_ID" ]; then
    export PROJECT_ID="$DEVSHELL_PROJECT_ID"
else
    export PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
fi

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "Warning: No project ID detected. Please run ./init.sh first."
    return 1 2>/dev/null || exit 1
fi

# Ensure gcloud config has project set
gcloud config set project "$PROJECT_ID" --quiet >/dev/null 2>&1

# 2. Export Vertex AI and ADK Environment Variables
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_GENAI_USE_VERTEXAI="TRUE"
export REGION="us-central1"
export GOOGLE_CLOUD_LOCATION="us-central1"
export MODEL_ID="${MODEL_ID:-gemini-3.7-flash}"


# 3. Set Application Default Credentials quota project
gcloud auth application-default set-quota-project "$PROJECT_ID" --quiet >/dev/null 2>&1 || true

# 4. Write to .env file in the repository root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cat <<ENVEOF > "$REPO_DIR/.env"
PROJECT_ID=$PROJECT_ID
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_GENAI_USE_VERTEXAI=TRUE
REGION=us-central1
GOOGLE_CLOUD_LOCATION=us-central1
MODEL_ID=$MODEL_ID
ENVEOF

echo "✓ Exported PROJECT_ID=$PROJECT_ID"
echo "✓ Exported GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo "✓ Exported GOOGLE_GENAI_USE_VERTEXAI=TRUE"
echo "✓ Exported GOOGLE_CLOUD_LOCATION=us-central1"
echo "✓ Exported MODEL_ID=$MODEL_ID"

echo "✓ Generated $REPO_DIR/.env"
