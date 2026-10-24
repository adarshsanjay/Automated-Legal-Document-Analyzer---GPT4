import re
from pathlib import Path
from datetime import datetime
from typing import List

import ollama
from pypdf import PdfReader

MODEL = "qwen3:4b"
MISSING = "Not mentioned in the document."


def extract_text_from_file(file_obj, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        content = file_obj.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)

    if suffix == ".pdf":
        reader = PdfReader(file_obj)
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not text:
            raise ValueError(
                "No selectable text found in PDF. OCR is not included."
            )

        return text

    raise ValueError("Please upload a .txt or .pdf file.")


def preprocess_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    return "".join(
        char for char in text
        if char in "\n\t" or ord(char) >= 32
    ).strip()


def split_into_chunks(text: str, max_chars: int = 20000) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        pieces = [
            paragraph[i:i + max_chars]
            for i in range(0, len(paragraph), max_chars)
        ]

        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}".strip()

    if current:
        chunks.append(current)

    return chunks


def ask_model(prompt: str) -> str:
    try:
        print("\n--- PROMPT SENT TO QWEN ---\n")
        print(prompt[:3000])
        print("\n--- END OF PROMPT ---\n")
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal document analysis assistant for "
                        "educational use. Do not provide legal advice. "
                        "Never invent facts, parties, dates, amounts, or "
                        "clauses. If information is absent from the complete "
                        f"document, state exactly: '{MISSING}'. "
                        "Distinguish facts from uncertainty."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )

        return response.message.content.strip()

    except Exception as exc:
        raise RuntimeError(
            "Could not communicate with Ollama. Make sure Ollama is running "
            "and qwen3:4b is installed. Original error: " + str(exc)
        ) from exc


def analyze_legal_document(
    text: str,
    document_name: str = "Legal document",
    max_chars: int = 20000,
) -> str:

    text = preprocess_text(text)

    if not text:
        raise ValueError("The document is empty.")

    chunks = split_into_chunks(text, max_chars)
    partial_reports = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""
Analyze this legal document excerpt.

Document: {document_name}
Chunk: {index} of {len(chunks)}

Extract information supported by the text. Do not guess.
For information not found in this excerpt, write
"Not established in this excerpt" because other sections may contain it.

Provide:
1. Document title and type
2. Parties and their roles
3. Purpose
4. Obligations of each party
5. Payment terms, fees and deadlines
6. Duration, effective date, renewal and expiry
7. Termination conditions and notice periods
8. Important dates
9. Confidentiality and data protection
10. Intellectual property
11. Liability, indemnity and warranties
12. Dispute resolution and governing law
13. Other important clauses
14. Ambiguities or issues requiring human review

DOCUMENT TEXT:
<<<
{chunk}
>>>
"""
        partial_reports.append(ask_model(prompt))

    if len(partial_reports) == 1:
        findings = partial_reports[0]

    else:
        combined = "\n\n".join(
            f"CHUNK {i + 1} FINDINGS:\n{report}"
            for i, report in enumerate(partial_reports)
        )

        prompt = f"""
Consolidate the following findings into one accurate report for
"{document_name}".

Remove duplicate information. Do not invent facts.
If information is absent from the complete document, write exactly:
"{MISSING}".
If details conflict, describe the conflict without guessing.

Use these headings:

LEGAL DOCUMENT SUMMARY
Document Type:
Parties:
Purpose:
Obligations:
Payment Terms:
Duration:
Termination Conditions:
Important Dates:
Key Clauses:
Ambiguities / Items Requiring Review:
Concise Summary:

FINDINGS:
{combined}
"""
        findings = ask_model(prompt)

    return (
        "LEGAL DOCUMENT ANALYSIS\n"
        + "=" * 60 + "\n"
        + f"Document: {document_name}\n"
        + f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        + f"Model: {MODEL} (local Ollama)\n\n"
        + findings
        + "\n\nDISCLAIMER: This report is for educational purposes only, "
        "is not legal advice, and should be checked by a qualified "
        "legal professional."
    )


def save_report(report: str, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(filename).name

    if not filename.lower().endswith(".txt"):
        filename += ".txt"

    output_path = output_dir / filename
    output_path.write_text(report, encoding="utf-8")

    return output_path
