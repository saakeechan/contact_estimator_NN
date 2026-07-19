#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-contact_estimator_pytorch:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-contact_estimator_train}"

usage() {
    cat <<EOF
Usage: ./docker.sh <command>

Commands:
  build   Build the image.
  run     Create/start the container and open a shell.
  shell   Open a shell in the running container.
  logs    Follow container logs.
  stop    Stop the container.
  remove  Remove the stopped container.
  status  Show container status.
EOF
}

case "${1:-help}" in
    build)
        docker build -t "$IMAGE_NAME" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR"
        ;;
    run)
        if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
            docker run -d --gpus all --name "$CONTAINER_NAME" \
                --user "$(id -u):$(id -g)" \
                -v "$REPO_DIR:/workspace" -w /workspace \
                "$IMAGE_NAME" sleep infinity
        else
            docker start "$CONTAINER_NAME" >/dev/null
        fi
        docker exec -it "$CONTAINER_NAME" bash
        ;;
    shell)
        docker exec -it "$CONTAINER_NAME" bash
        ;;
    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;
    stop)
        docker stop "$CONTAINER_NAME"
        ;;
    remove)
        docker rm "$CONTAINER_NAME"
        ;;
    status)
        docker ps -a --filter "name=^/${CONTAINER_NAME}$"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
