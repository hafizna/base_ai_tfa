"""Read and render browser-test PDFs for visual review; no report authoring."""
from pathlib import Path
import json
import fitz

root = Path(__file__).resolve().parents[1] / "tmp" / "browser-validation"
out = root / "pdf-pages"
out.mkdir(exist_ok=True)
summary = []
for variant in ("before", "after"):
    with fitz.open(root / f"{variant}-report.pdf") as doc:
        images = 0
        for index, page in enumerate(doc):
            images += len(page.get_images(full=True))
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(out / f"{variant}-{index + 1}.png")
        summary.append({"variant": variant, "pages": len(doc), "embedded_images": images,
                        "text_characters": sum(len(page.get_text()) for page in doc)})
print(json.dumps(summary))
