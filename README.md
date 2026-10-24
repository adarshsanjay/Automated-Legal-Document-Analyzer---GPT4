# Experiment 5 – Automated Legal Document Analyzer using Ollama + Qwen3:4b

A Python + Streamlit application that analyzes legal contracts using the **locally running Ollama model `qwen3:4b`**. It uses no OpenAI or other cloud model API and requires no API key.

## Features
- Paste legal text or upload `.txt` / text-based `.pdf`.
- Preprocess whitespace and control characters.
- Chunk long documents and consolidate findings.
- Extract parties, purpose, obligations, payment terms, duration, termination, dates, confidentiality, intellectual property, and key clauses.
- Instructs the model not to invent information and to state `Not mentioned in the document.` for absent information.
- Save or download a text report.
- Optional contract documents from a Kaggle dataset.

## Project structure
```text
Experiment_5_Legal_Document_Analyzer/
├── app.py
├── analyzer.py
├── requirements.txt
├── README.md
├── .gitignore
├── sample_contract.txt
├── data/README.md
└── outputs/.gitkeep
```

## Setup on macOS
1. Open Ollama and check the model in Terminal:
   ```bash
   ollama list
   ```
   Confirm `qwen3:4b` is listed. If needed, run `ollama serve` in a terminal (if the Ollama app is not already running).

2. Open this project folder in VS Code and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run:
   ```bash
   streamlit run app.py
   ```
   Open the local address shown, usually `http://localhost:8501`.

5. Upload/paste a contract, click **Analyze Document**, then save or download the report. Start with `sample_contract.txt`.

## No API required
Inference is performed by the local Ollama server using `qwen3:4b`. No `.env`, API key, or cloud API account is needed. The model must already be installed and Ollama must be running.

## Kaggle dataset
This project supports contract documents downloaded from the Atticus Commercial Contract Dataset / CUAD:
- Kaggle: https://www.kaggle.com/theatticusproject
- CUAD information: https://www.atticusprojectai.org/cuad/
- Repository: https://github.com/The-Atticus-Project/cuad

Download the dataset through Kaggle, then put extracted `.txt` or `.pdf` files inside `data/`. The app searches this folder recursively. The dataset is not bundled because it is distributed separately and may be large. Check the current license and attribution terms before using or redistributing it.

Optional Kaggle CLI setup:
```bash
python -m pip install kaggle
kaggle datasets list -s "Atticus Commercial Contract Dataset"
```
Use the actual dataset slug returned by Kaggle:
```bash
kaggle datasets download -d OWNER/DATASET-SLUG --unzip -p data
```
Replace the placeholder with the current owner and slug from Kaggle.

## Experiment mapping
| Requirement | Implementation |
|---|---|
| Q1 Read legal document | Paste text, TXT upload, PDF upload |
| Q2 Display information | Streamlit UI and preview |
| Q3 Preprocess | `preprocess_text()` |
| Q4 Prompt generation | Prompts in `analyze_legal_document()` |
| Q5 Summary | Local Qwen analysis |
| Q6 Save report | `save_report()` writes to `outputs/` |
| Q7 Complete app | `app.py` integrates the workflow |

## Limitations
- Educational prototype only, not legal advice.
- The model may omit or misinterpret clauses; verify every finding against the source.
- Scanned PDF OCR is not included.
- Cross-references between distant sections can be missed in chunked documents.
- Local inference speed depends on your Mac's available memory and compute.

## Dataset citation
If using CUAD, cite: Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. “CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.” NeurIPS 2021 Datasets and Benchmarks Track.
