#!/bin/bash

echo "=============================="
echo "Downloading medical datasets"
echo "=============================="

ROOT_DIR=$(pwd)

mkdir -p ISIC Kvasir-SEG BUSI REFUGE

############################
# 1. Kvasir-SEG (公开直链)
############################
echo "[1/4] Downloading Kvasir-SEG..."

cd "$ROOT_DIR/Kvasir-SEG"

URL="https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"
ZIP_NAME="Kvasir-SEG.zip"

if [ ! -f "$ZIP_NAME" ]; then
    wget $URL -O $ZIP_NAME || curl -L $URL -o $ZIP_NAME
    unzip $ZIP_NAME
else
    echo "Kvasir-SEG already downloaded."
fi

############################
# 2. ISIC
############################
echo "[2/4] ISIC Dataset"

cd "$ROOT_DIR/ISIC"
echo "ISIC dataset requires registration."
echo "Please download manually from:"
echo "https://challenge.isic-archive.com/"
echo "Then place the extracted files into data/ISIC/"

############################
# 3. BUSI
############################
echo "[3/4] BUSI Dataset"

cd "$ROOT_DIR/BUSI"
echo "BUSI dataset requires manual download."
echo "Please download from Kaggle:"
echo "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"
echo "Then extract files into data/BUSI/"

############################
# 4. REFUGE
############################
echo "[4/4] REFUGE Dataset"

cd "$ROOT_DIR/REFUGE"
echo "REFUGE dataset requires registration."
echo "Please download from:"
echo "https://refuge.grand-challenge.org/"
echo "Then place files into data/REFUGE/"

echo "=============================="
echo "Dataset preparation finished."
echo "=============================="
