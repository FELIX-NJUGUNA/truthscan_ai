#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK resources to local folder
python -m nltk.downloader -d ./nltk_data punkt averaged_perceptron_tagger
