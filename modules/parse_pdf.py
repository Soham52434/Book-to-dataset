from pathlib import Path
from typing import List, Dict

def _ocr_image_pil(pil_img):
    try:
        import pytesseract
        return pytesseract.image_to_string(pil_img)
    except Exception:
        return ""

def parse_pdf_to_pages(pdf_path: str, ocr_if_needed: bool = False) -> List[Dict]:
    """
    Returns: list of dicts: {page_num, text, meta}
    Attempts text extraction with PyMuPDF, falls back to OCR if page has very low text and OCR enabled.
    """
    import fitz  # PyMuPDF
    from PIL import Image
    import io

    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        meta = {"width": page.rect.width, "height": page.rect.height}
        # If page is mostly image and OCR requested
        if ocr_if_needed and len(text.strip()) < 20:
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = _ocr_image_pil(img) or text
            except Exception:
                pass
        pages.append({"page_num": i + 1, "text": text, "meta": meta})
    doc.close()
    return pages
