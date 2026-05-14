# `main_tag_embed_ROI.py` usage

This script generates synthetic full-tag images and Pascal VOC XML annotations for training the ROI/full-tag detector.

## Folder structure

Run the script from the folder that contains `main_tag_embed_ROI.py`.

Required input folders:

```text
roi-tag-generation/
├── main_tag_embed_ROI.py
├── scenes/
└── tags/
```

- `scenes/`: background images used for synthetic sample generation.
- `tags/`: four tag-piece images used to build the full tag. If files named `a`, `b`, `c`, and `d` are available, they are used directly. Otherwise, the first four files in alphabetical order are used.

## Main parameters

The main user-editable parameters are located near the beginning of the script.

| Parameter | Description |
|---|---|
| `SCENE_DIR` | Input folder containing background images. |
| `TAG_DIR` | Input folder containing the tag-piece images. |
| `OUTPUT_DIR` | Output folder where generated images, XML annotations, and CSV reports are saved. |
| `TRAIN_DIR` | Training output folder, created under `OUTPUT_DIR`. |
| `TEST_DIR` | Test output folder, created under `OUTPUT_DIR`. |
| `CROP_SIZE` | Background patch size used for each generated image. Default: `640`. |
| `CNN1_CLASS_NAME` | Class name written into XML annotations. Default: `ABCD`. |
| `COMPOSITE_SIZE_RANGES` | Full-tag size ranges used during synthetic data generation. |
| `COMPOSITE_TARGET_MAIN_TAGS_PER_BG` | Target number of full tags generated for each background image and size range. |
| `COMPOSITE_MIN_TAGS_PER_IMAGE` | Minimum number of full tags placed in one generated image. |
| `COMPOSITE_MAX_TAGS_PER_IMAGE` | Maximum number of full tags placed in one generated image. |
| `MAX_TRIES_PER_RANGE` | Maximum failed-placement attempts allowed for each size range. |
| `PIECE_SIZE_PX` | Size of each tag piece before the full tag is assembled. |
| `PIECE_GAP_PX` | Gap between tag pieces in the assembled full tag. |
| `ENABLE_JPEG_COMPAT` | Applies JPEG encode/decode simulation when enabled. |
| `JPEG_QUALITY_RANGE` | Random JPEG quality range used when JPEG compatibility is enabled. |
| `ALPHA_FORCE_METHOD` | Alpha blending method. Supported values: `gauss`, `dist`, `morph`, `random`. |
| `PIPELINE_MODE` | Synthetic generation mode. Supported values: `full`, `naive_same_scale`, `no_bg_aware_block`, `no_appearance_realism_block`. |
| `DEBUG_DRAW` | Draws debug bounding boxes on generated images when enabled. |

## Output files

The script creates:

```text
OUTPUT_DIR/
├── train/
│   ├── *.jpg
│   └── *.xml
├── test/
│   ├── *.jpg
│   └── *.xml
├── train_report.csv
└── test_report.csv
```

Each XML file contains one class name, `ABCD`, and the bounding box of the generated full tag.

## Running the script

Activate the virtual environment first:

```bat
synthetic-env\Scripts\activate
```

Then run:

```bat
python main_tag_embed_ROI.py
```

On Linux/macOS:

```bash
source synthetic-env/bin/activate
python main_tag_embed_ROI.py
```

## Notes

- Use a short project path on Windows to avoid long-path installation or file-access issues.
- Update `OUTPUT_DIR` before running the script.
- Keep `SCENE_DIR` and `TAG_DIR` as relative folders unless a different folder structure is required.
- The generated dataset is intended for training the ROI/full-tag detector.
