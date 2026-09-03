#!/bin/bash

# --- Function for error handling ---
handle_error() {
  echo -e "\n\n*******************************************************"
  echo "Error: $1"
  echo "*******************************************************"
  exit 1
}

echo "=== Initializing Enterprise Agent Evaluation Environment ==="

# --- Part 1: Resolve or Create Google Cloud Project ID ---
PROJECT_FILE="$HOME/project_id.txt"
PROJECT_ID_SET=false

# 1. Check if project ID file already exists and points to a valid project
if [ -s "$PROJECT_FILE" ]; then
    EXISTING_PROJECT_ID=$(cat "$PROJECT_FILE" | tr -d '[:space:]')
    echo "Found existing project ID in $PROJECT_FILE: $EXISTING_PROJECT_ID"
    if gcloud projects describe "$EXISTING_PROJECT_ID" --quiet >/dev/null 2>&1; then
        echo "Project '$EXISTING_PROJECT_ID' successfully verified."
        FINAL_PROJECT_ID=$EXISTING_PROJECT_ID
        PROJECT_ID_SET=true
        gcloud config set project "$FINAL_PROJECT_ID" || handle_error "Failed to set active project."
    else
        echo "Warning: Project '$EXISTING_PROJECT_ID' from file does not exist or lacks permissions."
        rm -f "$PROJECT_FILE"
    fi
fi

# 2. Check Cloud Shell environment variable $DEVSHELL_PROJECT_ID
if [ "$PROJECT_ID_SET" = false ] && [ -n "$DEVSHELL_PROJECT_ID" ]; then
    if gcloud projects describe "$DEVSHELL_PROJECT_ID" --quiet >/dev/null 2>&1; then
        echo "Detected valid Cloud Shell project: $DEVSHELL_PROJECT_ID"
        FINAL_PROJECT_ID=$DEVSHELL_PROJECT_ID
        PROJECT_ID_SET=true
        gcloud config set project "$FINAL_PROJECT_ID" || handle_error "Failed to set active project."
        echo "$FINAL_PROJECT_ID" > "$PROJECT_FILE"
    fi
fi

# 3. Check active gcloud config
if [ "$PROJECT_ID_SET" = false ]; then
    CONFIG_PROJECT=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
    if [ -n "$CONFIG_PROJECT" ] && [ "$CONFIG_PROJECT" != "(unset)" ]; then
        if gcloud projects describe "$CONFIG_PROJECT" --quiet >/dev/null 2>&1; then
            echo "Detected active gcloud config project: $CONFIG_PROJECT"
            FINAL_PROJECT_ID=$CONFIG_PROJECT
            PROJECT_ID_SET=true
            echo "$FINAL_PROJECT_ID" > "$PROJECT_FILE"
        fi
    fi
fi

# 4. Check available projects list
if [ "$PROJECT_ID_SET" = false ]; then
    FIRST_PROJECT=$(gcloud projects list --format="value(projectId)" --limit=1 2>/dev/null | tr -d '[:space:]')
    if [ -n "$FIRST_PROJECT" ]; then
        echo "Found available GCP project: $FIRST_PROJECT"
        FINAL_PROJECT_ID=$FIRST_PROJECT
        PROJECT_ID_SET=true
        gcloud config set project "$FINAL_PROJECT_ID"
        echo "$FINAL_PROJECT_ID" > "$PROJECT_FILE"
    fi
fi

# 5. If still no project found, create one
if [ "$PROJECT_ID_SET" = false ]; then
    echo "--- Creating New Google Cloud Project ---"
    RANDOM_SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 6)
    DEFAULT_ID="eval-agent-${RANDOM_SUFFIX}"
    read -p "Enter project ID or press Enter to use default [$DEFAULT_ID]: " USER_CHOICE
    FINAL_PROJECT_ID="${USER_CHOICE:-$DEFAULT_ID}"
    echo "Creating project: $FINAL_PROJECT_ID..."
    gcloud projects create "$FINAL_PROJECT_ID" --quiet || handle_error "Failed to create project."
    gcloud config set project "$FINAL_PROJECT_ID"
    echo "$FINAL_PROJECT_ID" > "$PROJECT_FILE"
fi

echo "=== Active Google Cloud Project: $FINAL_PROJECT_ID ==="

# --- Part 2: Enable Required APIs ---
echo "--- Enabling Required Google Cloud APIs ---"
gcloud services enable aiplatform.googleapis.com --quiet

# --- Part 3: Source set_env.sh to generate .env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/set_env.sh" ]; then
    source "$SCRIPT_DIR/set_env.sh"
fi

echo "=== Environment Setup Complete! ==="
