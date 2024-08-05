import subprocess

# Push video to Nginx for streaming
subprocess.run([
    'ffmpeg',
    '-re', 
    '-i', 'data/processed/output_video.mp4',
    '-c:v', 'libx264',
    '-b:v', '500k',
    '-f', 'flv',
    'rtmp://localhost/live/stream'
])
