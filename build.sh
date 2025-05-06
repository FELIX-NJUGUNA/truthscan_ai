#!/bin/bash

# Exit immediately on error
set -e

# Install Python dependencies
pip install -r requirements.txt

# Create the nltk_data directory explicitly
mkdir -p nltk_data

# Download NLTK resources
python -m nltk.downloader punkt averaged_perceptron_tagger -d ./nltk_data