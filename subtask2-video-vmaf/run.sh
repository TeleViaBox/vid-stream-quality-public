#!/bin/bash

# pip install -r requirements.txt

# Start Nginx server
./src/stream/setup_nginx.sh

# Start video capture and preprocessing
python3 src/capture/preprocess.py &

# Start streaming
python3 src/stream/stream_hls_dash.py &
