# Virtual environment setup

This setup is sufficient to run the synthetic data generation codes after the repository files are downloaded.

## Windows

Open Command Prompt or PowerShell in the project folder and run:

```bat
python -m venv synthetic-env
synthetic-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Linux / macOS

Open a terminal in the project folder and run:

```bash
python3 -m venv synthetic-env
source synthetic-env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After these commands are completed, the synthetic data generation codes are ready to use.

To leave the virtual environment:

```bash
deactivate
```
