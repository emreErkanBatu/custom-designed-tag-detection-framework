# `main_tag_part_embed.py` usage

This script generates synthetic images and Pascal VOC XML annotations for training the sub-component detector.

The generation process has two stages:

1. **Individual sub-component generation**: places single tag pieces on background images.
2. **Full-tag composite generation**: builds a full tag from four pieces and saves each piece as a separate annotated object.

## Folder structure

Run the script from the folder that contains `main_tag_part_embed.py`.

Required input folders:

```text
sub-component-generation/
├── main_tag_part_embed.py
├── scenes/
└── tags/
```

- `scenes/`: background images used for synthetic sample generation.
- `tags/`: four tag-piece images. If files named `A`, `B`, `C`, and `D` are available, they are used directly. Otherwise, the first four files in alphabetical order are used.

## Main parameters

The main user-editable parameters are located near the beginning of the script.

| Parameter | Description |
|---|---|
| `SCENE_DIR` | Input folder containing background images. |
| `TAG_DIR` | Input folder containing tag-piece images. |
| `OUTPUT_DIR` | Output folder where generated images, XML annotations, and CSV reports are saved. |
| `TRAIN_DIR` | Training output folder, created under `OUTPUT_DIR`. |
| `TEST_DIR` | Test output folder, created under `OUTPUT_DIR`. |
| `CROP_SIZE` | Background patch size used for each generated image. Default: `512`. |
| `SIZE_RANGES` | Size ranges used for individual sub-component generation. |
| `COMPOSITE_SIZE_RANGES` | Full-tag size ranges used during composite generation. |
| `COMPOSITE_TARGET_MAIN_TAGS` | Target number of full-tag composites generated for each background image and size range. |
| `COMPOSITE_MIN_TAGS_PER_IMAGE` | Minimum number of full-tag composites placed in one generated image. |
| `COMPOSITE_MAX_TAGS_PER_IMAGE` | Maximum number of full-tag composites placed in one generated image. |
| `MAX_TRIES_PER_RANGE` | Maximum failed-placement attempts allowed for each size range. |
| `PIECE_SIZE_PX` | Size of each tag piece before the full-tag composite is assembled. |
| `PIECE_GAP_PX` | Gap between tag pieces in the assembled full-tag composite. |
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
├── test_report.csv
├── train_report_composite_only.csv
└── test_report_composite_only.csv
```

The XML annotations contain the tag-piece class names, such as `A`, `B`, `C`, and `D`, with separate bounding boxes for each visible sub-component.

## Running the script

Activate the virtual environment first.

Windows:

```bat
synthetic-env\Scripts\activate
python main_tag_part_embed.py
```

Linux/macOS:

```bash
source synthetic-env/bin/activate
python main_tag_part_embed.py
```

## Notes

- Update `OUTPUT_DIR` before running the script.
- Keep `SCENE_DIR` and `TAG_DIR` as relative folders unless a different folder structure is required.
- Use a short project path on Windows to avoid long-path installation or file-access issues.
- The generated dataset is intended for training the sub-component detector.
