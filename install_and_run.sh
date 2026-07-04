#!/bin/bash

echo "=============================================="
echo "Installing X-Ray Spectrum App (Offline Linux)"
echo "=============================================="

INSTALL_DIR="$HOME/miniconda_xray"

# 1. Install Miniconda (if not already installed)
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Installing Miniconda (Python 3.11)..."
    chmod +x Miniconda3-py311_23.11.0-2-Linux-x86_64.sh
    ./Miniconda3-py311_23.11.0-2-Linux-x86_64.sh -b -p "$INSTALL_DIR"
fi

# 2. Activate Conda base environment
echo "Activating Python environment..."
source "$INSTALL_DIR/bin/activate"

# Verify python version
python --version

# 3. Install wheels offline
echo "Installing dependencies from local wheels..."
pip install --no-index --find-links=wheels -r requirements.txt

# 4. Launch Streamlit
echo "Launching Streamlit App..."
streamlit run app.py
