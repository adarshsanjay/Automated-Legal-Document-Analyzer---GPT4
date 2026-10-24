from pathlib import Path
import streamlit as st
import ollama
from analyzer import analyze_legal_document, extract_text_from_file, preprocess_text, save_report, MODEL

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="Legal Document Analyzer", page_icon="⚖️", layout="wide")
st.title("⚖️ Automated Legal Document Analyzer")
st.caption("Experiment 5 | Local Ollama inference using qwen3:4b")

with st.sidebar:
    st.subheader("Local model")
    st.code(MODEL)
    if st.button("Check Ollama model"):
        try:
            available = [getattr(m, "model", "") or getattr(m, "name", "") for m in ollama.list().models]
            if any(MODEL in m for m in available):
                st.success(f"{MODEL} is available.")
            else:
                st.warning("Model not found. Run `ollama list` in Terminal.")
        except Exception as e:
            st.error(f"Ollama is not responding. Open Ollama or run `ollama serve`. {e}")

st.warning("Educational prototype, not legal advice. Review the report against the source contract.")

tab1, tab2, tab3 = st.tabs(["Analyze a document", "Kaggle dataset files", "About"])

with tab1:
    upload = st.file_uploader("Upload a legal document (.txt or .pdf)", type=["txt", "pdf"])
    pasted = st.text_area("Or paste legal text", height=220)
    name = st.text_input("Document name", value="legal_document")
    max_chars = st.slider("Chunk size (characters)", 8000, 60000, 30000, 2000)
    if st.button("Analyze Document", type="primary", use_container_width=True):
        try:
            if upload:
                raw = extract_text_from_file(upload, upload.name)
                source = upload.name
            else:
                raw, source = pasted, name.strip() or "legal_document"
            cleaned = preprocess_text(raw)
            if len(cleaned) < 30:
                st.error("Upload or enter a legal document containing more text.")
            else:
                with st.spinner(f"Analyzing locally with {MODEL}..."):
                    st.session_state["report"] = analyze_legal_document(cleaned, source, max_chars)
                st.session_state["report_name"] = Path(source).stem
                st.success("Analysis completed.")
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("report"):
        st.subheader("Generated analysis")
        st.text_area("Report", st.session_state["report"], height=480)
        default_name = st.session_state.get("report_name", "legal_analysis") + "_analysis.txt"
        out_name = st.text_input("Output filename", default_name)
        if st.button("Save Report"):
            saved = save_report(st.session_state["report"], out_name, OUTPUT_DIR)
            st.success(f"Saved to {saved.relative_to(ROOT)}")
        st.download_button("Download report", st.session_state["report"], file_name=default_name, mime="text/plain")

with tab2:
    st.write("Download contract text files from the Kaggle dataset and place .txt or .pdf files in `data/`.")
    st.markdown("[Kaggle: The Atticus Project](https://www.kaggle.com/theatticusproject)")
    candidates = sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".pdf"})
    if not candidates:
        st.info("No local dataset files found. See README.md for instructions.")
    else:
        selected = st.selectbox("Choose a contract", candidates, format_func=lambda p: str(p.relative_to(DATA_DIR)))
        if st.button("Load dataset contract"):
            try:
                st.session_state["dataset_text"] = extract_text_from_file(selected, selected.name)
                st.session_state["dataset_name"] = selected.name
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("dataset_text"):
            st.text_area("Preview", st.session_state["dataset_text"][:10000], height=220)
            if st.button("Analyze dataset contract", type="primary"):
                try:
                    with st.spinner(f"Analyzing locally with {MODEL}..."):
                        st.session_state["report"] = analyze_legal_document(
                            st.session_state["dataset_text"], st.session_state.get("dataset_name", "dataset_contract")
                        )
                    st.session_state["report_name"] = Path(st.session_state.get("dataset_name", "dataset_contract")).stem
                    st.success("Analysis completed. View the report in the first tab.")
                except Exception as e:
                    st.error(str(e))

with tab3:
    st.markdown("""
### Workflow
Input → Preprocessing → Prompt generation → Local Qwen analysis → Structured report → Save report

### Supported inputs
- Keyboard/pasted text
- `.txt` files
- Text-based `.pdf` files (scanned PDF OCR is not included)

### Exit
Stop Streamlit using `Ctrl+C` in the terminal.
""")
