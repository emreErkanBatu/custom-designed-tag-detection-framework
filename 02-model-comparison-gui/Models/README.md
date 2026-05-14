# Model files

The trained model files are not included in this GitHub repository because of their file size.

Download the trained model archive from Zenodo:

```text
https://doi.org/10.5281/zenodo.20159467
```

After downloading the archive, extract it into the `model-comparison-gui/` directory.

The final folder structure should be:

```text
model-comparison-gui/
└── Models/
    ├── M1/
    ├── M2/
    ├── M4/
    └── M5/
```

## Model folders

- `M1`: previous ROI model
- `M2`: previous sub-component model
- `M5`: proposed ROI model
- `M4`: proposed sub-component model

Do not rename the model folders or files. The PyQt application uses these paths directly.

## Notes

- The model archive must be extracted before running the PyQt video comparison application.
- If the model files are missing, the application may start, but inference cannot be performed.
- Large trained model files are distributed through Zenodo, not GitHub.
