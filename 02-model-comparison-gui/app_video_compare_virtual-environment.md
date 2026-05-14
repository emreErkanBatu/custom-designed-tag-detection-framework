# Virtual environment setup

This guide provides two setup options for the PyQt video comparison application.

- Use the **CPU setup** for a simple installation that works on most systems.
- Use the **Windows GPU setup** only if an NVIDIA GPU, CUDA 11.2, cuDNN 8.1, and Python 3.10 are available.

## Option 1: CPU setup

Open Command Prompt or PowerShell in the application folder and run:

```bat
python -m venv gui-env
gui-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app_video_compare.py
```

## Option 2: Windows GPU setup

This option is intended for native Windows systems using TensorFlow 2.10.1 with NVIDIA GPU support.

Before installing the Python packages, make sure that the following components are installed:

```text
Python 3.10
CUDA 11.2
cuDNN 8.1
NVIDIA GPU driver
```

Create and activate the GPU environment:

```bat
py -3.10 -m venv gui-gpu-env
gui-gpu-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-gpu-windows.txt
```

Check whether TensorFlow can detect the GPU:

```bat
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If the GPU is detected, run the application:

```bat
python app_video_compare.py
```

A valid GPU setup should return a non-empty GPU list, for example:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Optional: reduce TensorFlow console messages

```bat
set TF_CPP_MIN_LOG_LEVEL=2
python app_video_compare.py
```

## Notes

- Use a short project path on Windows to avoid long-path installation or file-access issues.
- Do not use the CPU and GPU environments at the same time.
- If the command prompt shows both `(base)` and the virtual environment name, run `conda deactivate` before activating the project environment.
- Native Windows GPU support is intended for TensorFlow 2.10.1. Newer TensorFlow versions should be used with WSL2 for GPU acceleration.
