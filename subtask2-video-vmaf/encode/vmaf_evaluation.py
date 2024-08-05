#!/bin/bash
# Script to evaluate video quality using VMAF
ffmpeg -i data/raw/input_video.mp4 -i data/processed/output_video.mp4 -lavfi libvmaf="log_path=data/logs/vmaf.json" -f null -
