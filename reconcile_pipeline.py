from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import ollama


PDF_DIR = Path("1_inputs/pdfs")
DRAFT_MD_DIR = Path("1_inputs/draft_mds")
OUTPUT_DIR = Path("2_outputs/final_vault_mds")
MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are an expert data reconciliation engineer working with clinical "
    "documents. I will provide you with 'Source of Truth Text' (which has "
    "the correct paragraph reading order but no formatting) and 'Draft "
    "Markdown' (which has formatting, tables, and visual element tags, but "
    "may have dropped text or bad column ordering). YOUR TASK: 1. Output "
    "valid Markdown. 2. Use the 'Source of Truth Text' as your strict guide "
    "for the text content and paragraph reading order. DO NOT drop any "
    "sentences from the Source of Truth. 3. Insert the markdown tables, "
    "bolding, headings, and `> **[VISUAL ELEMENT DETECTED]**` or "
    "`![Visual Element]` tags from the 'Draft Markdown' into their correct "
    "locations within the text. 4. Do not add any conversational filler. "
    "Output ONLY the final merged Markdown. "
    "CRITICAL RULES: "
    "1. Do NOT invent new image file names or URLs. However, if the "
    "'Draft Markdown' contains an exact image link (e.g., "
    "`![Visual Element](images/...)`), you MUST copy that exact markdown "
    "link verbatim into the final output at the appropriate location. "
    "Do not alter the URL path. "
    "2. Do NOT place visual tags inside individual table cells or rows; "
    "if a table has an associated visual tag, place it exactly once above "
    "or below the table. "
    "3. NEVER start your response with 'Here is the final merged Markdown' "
    "or any conversational filler. Output the raw Markdown directly."
)


def clean_raw_pdf_text(raw_text: str) -> str:
    """
    Remove hyphenated line breaks and reflow soft line breaks from PyMuPDF text.
    Double newlines (paragraph breaks) are preserved.
    """
    if not raw_text:
        return raw_text
    t = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"-\n\s*", "", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r" +", " ", t)
    return t.strip()


def strip_llm_chatty_output(content: str) -> str:
    """Remove common conversational prefixes from model output."""
    if not content:
        return content
    s = content.strip()

    intro_patterns = [
        r"(?is)^\s*(?:sure[!.,]?\s*|certainly[!.,]?\s*|of course[!.,]?\s*|"
        r"absolutely[!.,]?\s*|okay[!.,]?\s*|ok[!.,]?\s*)+(?:\n\s*)*",
        r"(?is)^\s*here(?:'s| is) the final merged markdown\s*:?\s*"
        r"(?:\n\s*[-—]{2,}\s*)?(?:\n\s*)*",
        r"(?is)^\s*below is the final merged markdown\s*:?\s*"
        r"(?:\n\s*[-—]{2,}\s*)?(?:\n\s*)*",
        r"(?is)^\s*the following is the final merged markdown\s*:?\s*"
        r"(?:\n\s*[-—]{2,}\s*)?(?:\n\s*)*",
        r"(?is)^\s*final merged markdown\s*:?\s*(?:\n\s*[-—]{2,}\s*)?(?:\n\s*)*",
    ]
    for pat in intro_patterns:
        prev = None
        while prev != s:
            prev = s
            s = re.sub(pat, "", s, count=1).strip()

    # Drop only leading separator lines (e.g. after "Here is...:\n---"), not HRs in body text.
    s = re.sub(r"(?s)^(?:\s*[-—]{3,}\s*\n)+", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def ensure_directories() -> None:
    """Ensure required input/output directories exist."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_MD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    """Normalize names for robust matching."""
    lowered = name.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(cleaned.split())


def first_words_key(name: str, n: int = 3) -> str:
    """Take the first N normalized words as a matching key."""
    words = normalize_name(name).split()
    return " ".join(words[:n])


def similarity_score(pdf_stem: str, md_stem: str) -> float:
    """Compute combined score using prefix-word and fuzzy similarity."""
    pdf_norm = normalize_name(pdf_stem)
    md_norm = normalize_name(md_stem)
    if not pdf_norm or not md_norm:
        return 0.0

    seq_score = difflib.SequenceMatcher(None, pdf_norm, md_norm).ratio()

    # Bonus if first 2-3 words overlap heavily.
    key2_pdf = first_words_key(pdf_stem, n=2)
    key2_md = first_words_key(md_stem, n=2)
    key3_pdf = first_words_key(pdf_stem, n=3)
    key3_md = first_words_key(md_stem, n=3)
    prefix_bonus = 0.0
    if key2_pdf and key2_pdf == key2_md:
        prefix_bonus += 0.2
    if key3_pdf and key3_pdf == key3_md:
        prefix_bonus += 0.2

    return min(1.0, seq_score + prefix_bonus)


def match_pdf_to_md(pdf_path: Path, md_candidates: List[Path]) -> Optional[Path]:
    """Pick the best matching draft markdown for a PDF."""
    if not md_candidates:
        return None

    scored: List[Tuple[float, Path]] = []
    for md_path in md_candidates:
        score = similarity_score(pdf_path.stem, md_path.stem)
        scored.append((score, md_path))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_md = scored[0]

    # Guardrail threshold keeps accidental pairings lower risk.
    return best_md if best_score >= 0.45 else None


def parse_markdown_pages(markdown_text: str) -> Dict[int, str]:
    """
    Parse markdown split by headers formatted exactly as `## Page X`.
    Returns {page_number: page_content}.
    """
    pattern = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown_text))
    pages: Dict[int, str] = {}

    if not matches:
        # Fallback: treat whole markdown as page 1 if markers are missing.
        return {1: markdown_text.strip()}

    for idx, match in enumerate(matches):
        page_number = int(match.group(1))
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        page_content = markdown_text[content_start:content_end].strip()
        pages[page_number] = page_content

    return pages


def reconcile_page(raw_pdf_text: str, draft_page_md: str, page_number: int) -> str:
    """Call Ollama to reconcile one page."""
    source_text = clean_raw_pdf_text(raw_pdf_text)
    user_prompt = (
        f"Page Number: {page_number}\n\n"
        "Raw PDF Text (Source of Truth):\n"
        "-----\n"
        f"{source_text}\n"
        "-----\n\n"
        "Draft Markdown:\n"
        "-----\n"
        f"{draft_page_md.strip()}\n"
        "-----\n\n"
        "Return only the final merged Markdown for this page."
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )
    raw_content = (response.get("message", {}).get("content") or "").strip()
    return strip_llm_chatty_output(raw_content)


def build_output_filename(pdf_path: Path, md_path: Path) -> str:
    """
    Build output filename as <base_matched_name>_final.md.
    Uses markdown base as the matched-name anchor.
    """
    base_name = md_path.stem.strip() or pdf_path.stem.strip()
    safe_base = re.sub(r"[\\/:*?\"<>|]+", "_", base_name)
    return f"{safe_base}_final.md"


def process_single_pair(pdf_path: Path, md_path: Path) -> None:
    """Process one matched PDF/MD pair."""
    output_name = build_output_filename(pdf_path, md_path)
    output_path = OUTPUT_DIR / output_name

    if output_path.exists():
        print(f"[SKIP] Already processed: {output_path.name}")
        return

    print(f"[START] PDF: {pdf_path.name}")
    print(f"        Draft MD: {md_path.name}")
    print(f"        Output: {output_path.name}")

    draft_text = md_path.read_text(encoding="utf-8", errors="ignore")
    md_pages = parse_markdown_pages(draft_text)
    if not md_pages:
        print(f"[WARN] No page blocks found in: {md_path.name}. Skipping.")
        return

    reconciled_pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        total_pdf_pages = len(doc)
        for page_number in sorted(md_pages.keys()):
            page_index = page_number - 1
            draft_page_md = md_pages.get(page_number, "").strip()

            raw_pdf_text = ""
            if 0 <= page_index < total_pdf_pages:
                raw_pdf_text = doc[page_index].get_text("text").strip()
            else:
                print(
                    f"[WARN] Page {page_number} out of PDF range ({total_pdf_pages}) "
                    f"for {pdf_path.name}. Using empty source text."
                )

            if not raw_pdf_text and not draft_page_md:
                print(f"[PAGE {page_number}] Empty source + draft. Skipping page.")
                continue

            print(f"[PAGE {page_number}] Reconciling...")
            merged_md = reconcile_page(raw_pdf_text, draft_page_md, page_number)
            if not merged_md:
                # Keep fallback content if model fails to return text.
                print(
                    f"[PAGE {page_number}] Empty model response. "
                    "Falling back to draft markdown."
                )
                merged_md = draft_page_md

            reconciled_pages.append(f"## Page {page_number}\n\n{merged_md}".strip())

    if not reconciled_pages:
        print(f"[WARN] No reconciled pages produced for {pdf_path.name}. Skipping save.")
        return

    final_text = "\n\n".join(reconciled_pages).rstrip() + "\n"
    output_path.write_text(final_text, encoding="utf-8")
    print(f"[DONE] Saved: {output_path}")


def run_batch_reconciliation() -> None:
    """Batch process all PDFs with robust matching to draft markdown files."""
    ensure_directories()
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    md_files = sorted(DRAFT_MD_DIR.glob("*.md"))

    print(f"[INFO] PDFs found: {len(pdf_files)}")
    print(f"[INFO] Draft MDs found: {len(md_files)}")

    if not pdf_files:
        print(f"[INFO] No PDFs found in: {PDF_DIR}")
        return
    if not md_files:
        print(f"[INFO] No draft markdown files found in: {DRAFT_MD_DIR}")
        return

    for pdf_path in pdf_files:
        matched_md = match_pdf_to_md(pdf_path, md_files)
        if not matched_md:
            print(f"[WARN] No confident markdown match for PDF: {pdf_path.name}")
            continue

        try:
            process_single_pair(pdf_path, matched_md)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR] Failed processing PDF '{pdf_path.name}' "
                f"with MD '{matched_md.name}': {exc}"
            )
            # Continue with next file for resilient batch behavior.
            continue

    print("[COMPLETE] Batch reconciliation finished.")


if __name__ == "__main__":
    run_batch_reconciliation()
