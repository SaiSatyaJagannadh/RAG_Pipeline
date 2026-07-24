"""Checks upload path/category sanitization. Run: python test_upload.py"""
from pathlib import Path

from app.uploads import DATA_DIR, safe_dest

ROOT = Path(DATA_DIR)


def test():
    assert safe_dest("PTO Policy.pdf", "policies") == ROOT / "policies/PTO Policy.pdf"
    assert safe_dest("a.docx", "faqs") == ROOT / "faqs/a.docx"

    # traversal is stripped, not rejected -> must land inside DATA_DIR
    assert safe_dest("../../../etc/passwd.pdf", "policies") == ROOT / "policies/passwd.pdf"

    for bad_name in ["notes.exe", "slides.pptx", ".hidden.pdf", "", "nodots"]:
        try:
            safe_dest(bad_name, "policies")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad filename: {bad_name!r}")

    for bad_cat in ["../policies", "pol icies", "", "a" * 41, "pol/icies"]:
        try:
            safe_dest("a.pdf", bad_cat)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad category: {bad_cat!r}")

    print("ok")


if __name__ == "__main__":
    test()
