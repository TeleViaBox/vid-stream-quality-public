#!/bin/bash
# Script to encode video using FFmpeg
ffmpeg -i data/raw/input_video.mp4 -c:v libx264 -b:v 500k data/processed/output_video.mp4