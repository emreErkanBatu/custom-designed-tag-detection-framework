# Background-aware synthetic data generation and adaptive ROI updating

This repository provides the source codes and supporting files associated with the accepted manuscript on synthetic data generation and adaptive ROI-based object/sub-component detection of a custom-designed tag.

The main method is referred to as the **Background-Aware Synthetic Tag Filtering and Blending Strategy**. The repository is organized to support reproducibility, reuse, and independent testing of the proposed framework.

## Overview

The experimental framework consists of two main detection stages:

1. **ROI/full-tag detection** using Faster R-CNN + ResNet50
2. **Sub-component detection** using CenterNet + ResNet50

The synthetic data generation codes are separated according to these two stages. A PyQt-based comparison interface is also provided to test the trained models on video inputs.

Large files such as trained models and raw test videos are distributed through Zenodo rather than GitHub.

## Demonstration video

The following video shows the PyQt comparison interface running the trained models on real-world test videos. It illustrates the use of background-aware synthetic data generation and adaptive ROI updating for sub-component detection.

[![Background-aware synthetic data + adaptive ROI for sub-component detection](https://img.youtube.com/vi/CxVaIhwIyDY/sddefault.jpg)](https://youtu.be/CxVaIhwIyDY)

## Repository structure

```text
.
├── 01-synthetic-data-generation/
│   ├── requirements.txt
│   ├── virtual-environment-setup.md
│   ├── roi-tag-generation/
│   │   ├── main_tag_embed_ROI.py
│   │   ├── main_tag_embed_ROI_usage.md
│   │   ├── scenes/
│   │   └── tags/
│   └── sub-component-generation/
│       ├── main_tag_part_embed.py
│       ├── main_tag_part_embed_usage.md
│       ├── scenes/
│       └── tags/
│
└── 02-model-comparison-gui/
    ├── app_video_compare.py
    ├── score_panel_overlay.py
    ├── two_stage_infer_class.py
    ├── requirements.txt
    ├── requirements-gpu-windows.txt
    ├── app_video_compare_virtual-environment.md
    ├── Models/
    │   └── README.md
    └── test-videos/
        └── README.md
```

## Synthetic data generation

The synthetic data generation codes are located in:

```text
01-synthetic-data-generation/
```

This folder contains two separate pipelines.

### ROI/full-tag generation

```text
01-synthetic-data-generation/roi-tag-generation/
```

This pipeline generates synthetic full-tag images and Pascal VOC XML annotations for training the ROI/full-tag detector.

Main script:

```text
main_tag_embed_ROI.py
```

Usage details are provided in:

```text
main_tag_embed_ROI_usage.md
```

### Sub-component generation

```text
01-synthetic-data-generation/sub-component-generation/
```

This pipeline generates synthetic images and Pascal VOC XML annotations for training the sub-component detector.

Main script:

```text
main_tag_part_embed.py
```

Usage details are provided in:

```text
main_tag_part_embed_usage.md
```

## Synthetic data generation setup

From the `01-synthetic-data-generation/` folder, create and activate a virtual environment.

### Windows

```bat
python -m venv synthetic-env
synthetic-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv synthetic-env
source synthetic-env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After installation, run the relevant generation script from its own folder.

## Model comparison GUI

The PyQt-based comparison application is located in:

```text
02-model-comparison-gui/
```

This interface allows testing the trained models on video files and comparing the previous approach with the proposed framework.

Main script:

```text
app_video_compare.py
```

## Trained models

The trained model files are not included in this GitHub repository because of their file size.

They are available through Zenodo:

```text
https://doi.org/10.5281/zenodo.20159467
```

After downloading the model archive, extract it into:

```text
02-model-comparison-gui/
```

The final folder structure should be:

```text
02-model-comparison-gui/
└── Models/
    ├── M1/
    ├── M2/
    ├── M4/
    └── M5/
```

Model folder definitions:

- `M1`: previous ROI model
- `M2`: previous sub-component model
- `M5`: proposed ROI model
- `M4`: proposed sub-component model

Do not rename the model folders or files. The PyQt application uses these paths directly.

## Raw test videos

The raw test videos are not included in this GitHub repository because of their file size.

They are available through Zenodo:

```text
https://doi.org/10.5281/zenodo.20160270
```

The video archive contains:

```text
Env_01.mp4
Env_2.mp4
Env_3.mp4
Env_4.mp4
Env_5.mp4
Env_6a.mp4
Env_6b.mp4
```

`Env_6a.mp4` and `Env_6b.mp4` are separate recordings from Environment-6.

The videos can be extracted to any local folder and selected directly from the PyQt application.

## Model comparison GUI setup

### CPU setup

From the `02-model-comparison-gui/` folder:

```bat
python -m venv gui-env
gui-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app_video_compare.py
```

### Optional Windows GPU setup

Native Windows GPU support requires TensorFlow 2.10.1 with Python 3.10, CUDA 11.2, and cuDNN 8.1.

```bat
conda create -n gui-gpu-env python=3.10 -y
conda activate gui-gpu-env
python -m pip install --upgrade pip
pip install -r requirements-gpu-windows.txt
python app_video_compare.py
```

To check whether TensorFlow detects the GPU:

```bat
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If the GPU is detected correctly, the output should include a GPU device.

## Notes for Windows users

Use a short project path, such as:

```text
F:\GitHub\tag-detection-framework
```

Long paths may cause package installation or file-access issues on Windows.

If the command prompt shows both `(base)` and the virtual environment name, deactivate Conda before activating the project environment:

```bat
conda deactivate
```

## Data and model availability

Large files are distributed through Zenodo:

| Resource | DOI |
|---|---|
| Trained models | https://doi.org/10.5281/zenodo.20159467 |
| Raw test videos | https://doi.org/10.5281/zenodo.20160270 |

The GitHub repository contains the source codes, setup files, usage instructions, and lightweight example resources.

## Citation

If you use this repository, the trained models, or the test videos, please cite the associated manuscript and the relevant Zenodo records.

```bibtex
@misc{erkan_trained_models_2026,
  author       = {Erkan, Emre},
  title        = {Trained models for Background-Aware Synthetic Tag Filtering and Blending Strategy},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20159467},
  url          = {https://doi.org/10.5281/zenodo.20159467}
}

@misc{erkan_raw_test_videos_2026,
  author       = {Erkan, Emre},
  title        = {Raw test videos for evaluating the proposed detection framework across real-world environments},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20160270},
  url          = {https://doi.org/10.5281/zenodo.20160270}
}
```

The article citation will be added after the final publication information becomes available.

## License

The source codes and documentation in this repository are provided for academic and non-commercial research use only.

Commercial use requires prior written permission from the author.

See the `LICENSE` file for details.

The trained models and raw test videos distributed through Zenodo are governed by the licenses specified in their corresponding Zenodo records.
