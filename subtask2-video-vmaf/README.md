

# project launch script

- run.sh
```
#!/bin/bash

pip install -r requirements.txt

# Start the Nginx server
./src/stream/setup_nginx.sh

# Start video capture and preprocessing
python src/capture/capture_video.py &

# Start adaptive bitrate streaming
python src/stream/stream_hls_dash.py
```

# for the pretrained model
```
conda activate leos-vehi-net-task2
pip install torch torchvision
pip install pillow
pip install git+https://github.com/xinntao/Real-ESRGAN.git
```


# Final goal (in progress)

```
video_streaming_project/
│
├── data/                             # Directory for storing data files
│   ├── raw/                          # Original video files
│   └── processed/                    # Preprocessed video files
│
├── models/                           # Directory for storing deep learning models
│   └── super_resolution_model.h5     # Pre-trained super-resolution model
│
├── logs/                             # Directory for storing log files and evaluation results
│   ├── vmaf.json                     # VMAF evaluation results log
│   └── training_logs.log             # Model training logs
│
├── src/                              # Source code directory
│   ├── capture/                      # Video capture and preprocessing module
│   │   ├── capture_video.py          # Video capture with OpenCV
│   │   └── preprocess.py             # Video preprocessing with deep learning
│   │
│   ├── encode/                       # Video encoding and quality evaluation module
│   │   ├── encode_video.py           # Video encoding with FFmpeg
│   │   └── vmaf_evaluation.py        # VMAF quality evaluation with libvmaf
│   │
│   ├── stream/                       # Adaptive bitrate streaming module
│   │   ├── setup_nginx.sh            # Configure Nginx RTMP server
│   │   ├── stream_hls_dash.py        # Streaming with DASH/HLS protocols
│   │   └── bitrate_adjustment.py     # Bitrate adjustment based on VMAF
│   │
│   ├── client/                       # Client-side decoding and playback module
│   │   ├── html5_player.html         # HTML5 video player page
│   │   └── vlc_player.py             # Video streaming with VLC player
│   │
│   └── ml/                           # Machine learning and prediction model module
│       ├── bandwidth_prediction.py   # Bandwidth prediction model
│       └── train_model.py            # Model training script
│
├── config/                           # Configuration files directory
│   ├── nginx.conf                    # Nginx server configuration file
│   └── model_config.json             # Deep learning model configuration
│
├── requirements.txt                  # Python dependencies list
├── README.md                         # Project documentation
└── run.sh                            # Script to start the project

```