# Phase 0 Extraction Pipeline

## Setup

1. Create and activate the virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Populate `.env` with `AZURE_PAT`, `AZURE_ORG`, and `AZURE_PROJECT`.

## Run

Extract bugs:

```powershell
python src/extract_bugs.py
```

Extract test cases:

```powershell
python src/extract_testcases.py
```

Merge the master dataset and write validation output:

```powershell
python src/merge_dataset.py
```

## Outputs

- `data/bugs.csv`
- `data/testcases.csv`
- `data/master_dataset.csv`
- `data/validation_report.txt`