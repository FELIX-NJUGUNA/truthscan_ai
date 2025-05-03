#!/bin/bash

# Download NLTK resources into a local directory
python -m nltk.downloader -d nltk_data punkt averaged_perceptron_tagger
