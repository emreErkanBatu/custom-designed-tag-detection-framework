# Test videos

The raw test videos are not included in this GitHub repository because of their file size.

The videos are distributed through Zenodo and can be used with the PyQt video comparison application.

Zenodo record:

```text
https://doi.org/10.5281/zenodo.20160270
```

## Video files

The video archive contains the following files:

```text
Env_01.mp4
Env_2.mp4
Env_3.mp4
Env_4.mp4
Env_5.mp4
Env_6a.mp4
Env_6b.mp4
```

These files correspond to the real-world test environments used for visual evaluation with the PyQt application. `Env_6a.mp4` and `Env_6b.mp4` are separate recordings from Environment-6.

## Expected usage

After downloading the video archive from Zenodo, extract the videos to any local folder and select the desired video from the PyQt application.

The videos do not have to be placed inside the GitHub repository. However, if you prefer to keep them inside the project folder, the following structure can be used:

```text
model-comparison-gui/
└── test-videos/
    ├── Env_01.mp4
    ├── Env_2.mp4
    ├── Env_3.mp4
    ├── Env_4.mp4
    ├── Env_5.mp4
    ├── Env_6a.mp4
    └── Env_6b.mp4
```

## Notes

- Large video files are distributed through Zenodo, not GitHub.
- The PyQt application allows selecting videos from any local folder.
- If a video appears rotated in OpenCV-based applications, the file may contain orientation metadata. In that case, use a corrected video file or re-encode the video with the correct physical orientation.
