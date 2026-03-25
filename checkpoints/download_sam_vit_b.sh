#!/bin/bash

echo "Downloading SAM ViT-B checkpoint..."

URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
FILENAME="sam_vit_b_01ec64.pth"

# if exist -> skip
if [ -f "$FILENAME" ]; then
    echo "Checkpoint already exists: $FILENAME"
    exit 0
fi

# wget or curl
if command -v wget &> /dev/null; then
    wget $URL -O $FILENAME
elif command -v curl &> /dev/null; then
    curl -L $URL -o $FILENAME
else
    echo "Error: neither wget nor curl is installed."
    exit 1
fi

echo "Download finished: $FILENAME"
