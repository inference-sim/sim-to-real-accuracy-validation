#!/bin/bash
# Start LLMServingSim Docker container
# This script replaces docker-compose for simpler setup

set -e

CONTAINER_NAME="llmservingsim-container"
IMAGE="astrasim/tutorial-micro2024"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running."
    echo "Please start Docker Desktop from Applications and try again."
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' already exists."

    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running."
    else
        echo "Starting existing container..."
        docker start "${CONTAINER_NAME}"
        echo "Container started successfully."
    fi
else
    echo "Creating new container '${CONTAINER_NAME}'..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -v "$(pwd)/LLMServingSim:/app/LLMServingSim" \
        -v "$(pwd)/vllm_data:/app/vllm_data:ro" \
        -v /tmp:/tmp \
        -w /app/LLMServingSim \
        "${IMAGE}" \
        tail -f /dev/null

    echo "Container created and started successfully."
fi

echo ""
echo "✅ Container '${CONTAINER_NAME}' is ready!"
echo ""
echo "Next steps:"
echo "  1. Run experiments: python -m experiment.run --adapters llmservingsim"
echo "  2. The adapter will automatically build LLMServingSim on first run (~5-10 min)"
echo ""
echo "Container management:"
echo "  - Stop:    docker stop ${CONTAINER_NAME}"
echo "  - Start:   docker start ${CONTAINER_NAME}"
echo "  - Shell:   docker exec -it ${CONTAINER_NAME} bash"
echo "  - Remove:  docker rm ${CONTAINER_NAME}"
