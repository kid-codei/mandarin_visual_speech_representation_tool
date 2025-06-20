#!/bin/bash

# install Homebrew if not installed
which -s brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# install packages
brew install ffmpeg espeak

# set up python venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt