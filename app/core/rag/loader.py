"""Document loading utilities — PDF, DOCX, and plain text/Markdown.

Only PDF has real page boundaries; DOCX/TXT/MD are returned as a single
"page 1" since page numbers there are a rendering artifact, not a property
of the source file. Citations for those formats degrade to "page 1" —
documented limitation, not a bug.
"""

from docx import Document as DocxDocument
from pypdf import PdfReader


def load_pdf_pages(file_path: str) -> list[tuple[int, str]]:
    """Extract text from a PDF, page by page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List of (page_number, text) tuples, 1-indexed, skipping blank pages.
    """
    reader = PdfReader(file_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def load_docx_pages(file_path: str) -> list[tuple[int, str]]:
    """Extract text from a Word document as a single page.

    Args:
        file_path: Path to the .docx file on disk.

    Returns:
        ``[(1, full_text)]``, or ``[]`` if the document has no text.
    """
    doc = DocxDocument(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(1, text)] if text.strip() else []


def load_text_pages(file_path: str) -> list[tuple[int, str]]:
    """Read a plain text or Markdown file as a single page.

    Args:
        file_path: Path to the .txt/.md file on disk.

    Returns:
        ``[(1, full_text)]``, or ``[]`` if the file is empty.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [(1, text)] if text.strip() else []
