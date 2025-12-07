# FFmpeg Installation Guide

FFmpeg is required for generating MP4 videos from the spectral zoom video generator. Without FFmpeg, the script will still generate individual frames that can be manually assembled into a video.

## Windows Installation

### Option 1: Using Chocolatey (Recommended)

If you have [Chocolatey](https://chocolatey.org/) installed:

```powershell
choco install ffmpeg
```

### Option 2: Manual Installation

1. Download FFmpeg from the official website:
   - Visit: https://ffmpeg.org/download.html
   - Click "Windows" → "Windows builds from gyan.dev"
   - Download the "ffmpeg-release-essentials.zip"

2. Extract the ZIP file to a permanent location:
   ```
   C:\ffmpeg\
   ```

3. Add FFmpeg to your PATH:
   - Open System Properties → Advanced → Environment Variables
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK on all dialogs

4. Verify installation:
   ```powershell
   ffmpeg -version
   ```

### Option 3: Using Scoop

If you have [Scoop](https://scoop.sh/) installed:

```powershell
scoop install ffmpeg
```

## Linux Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

### Fedora/RHEL
```bash
sudo dnf install ffmpeg
```

### Arch Linux
```bash
sudo pacman -S ffmpeg
```

## macOS Installation

### Using Homebrew (Recommended)
```bash
brew install ffmpeg
```

### Using MacPorts
```bash
sudo port install ffmpeg
```

## Verification

After installation, verify FFmpeg is working:

```bash
ffmpeg -version
```

You should see output like:
```
ffmpeg version 6.x.x Copyright (c) 2000-2024 the FFmpeg developers
built with gcc ...
```

## Usage with Pixel Maxwell Demon

Once FFmpeg is installed, the video generation scripts will automatically detect it:

```bash
# Generate spectral zoom video with MP4 output
python generate_spectral_zoom_video.py

# Validate motion picture demon with MP4 output
python validate_motion_picture_demon.py

# Multi-modal motion picture with MP4 output
python validate_multi_modal_motion_picture.py
```

## Alternative: Manual Video Assembly

If you cannot install FFmpeg, the scripts will save individual frames. You can assemble them manually:

### Using FFmpeg (after installation)
```bash
cd spectral_zoom_video/video_frames
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p ../spectral_temporal_zoom.mp4
```

### Using Online Tools
- Upload frames to online video makers like:
  - ezgif.com
  - gifmaker.me
  - Any "images to video" converter

### Using Python (Alternative)
If you have OpenCV installed:

```python
import cv2
import numpy as np
from pathlib import Path

frames_dir = Path('spectral_zoom_video/video_frames')
frame_files = sorted(frames_dir.glob('frame_*.png'))

# Read first frame to get dimensions
first_frame = cv2.imread(str(frame_files[0]))
height, width = first_frame.shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('spectral_temporal_zoom.mp4', fourcc, 30, (width, height))

# Write all frames
for frame_file in frame_files:
    frame = cv2.imread(str(frame_file))
    out.write(frame)

out.release()
print("Video created!")
```

## Troubleshooting

### "FFmpeg not found" error
- Ensure FFmpeg is in your PATH
- Restart your terminal/PowerShell after installation
- Try running `ffmpeg -version` to verify

### "Could not find codec" error
- Reinstall FFmpeg with full codecs (not "essentials" build)
- On Linux, install `ffmpeg` AND `libx264`

### Permission errors
- Run terminal/PowerShell as administrator (Windows)
- Use `sudo` on Linux/macOS

### Path issues on Windows
- Use forward slashes: `C:/ffmpeg/bin`
- Or escape backslashes: `C:\\ffmpeg\\bin`
- Restart PowerShell after adding to PATH

## License Note

FFmpeg is licensed under LGPL 2.1+. Make sure to comply with the license if distributing binaries.

## Additional Resources

- Official FFmpeg documentation: https://ffmpeg.org/documentation.html
- FFmpeg wiki: https://trac.ffmpeg.org/wiki
- FFmpeg command line examples: https://ffmpeg.org/ffmpeg.html

