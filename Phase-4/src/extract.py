import re

import fitz
from docx import Document


def normalize_extracted_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(
        r"-\n",
        "",
        text
    )
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )
    text = re.sub(
        r"\n +",
        "\n",
        text
    )
    return text.strip()


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_docx(file_path):
    doc = Document(file_path)

    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


def read_pdf(file_path):
    pdf = fitz.open(file_path)

    text = ""

    for page in pdf:
        blocks = page.get_text("blocks")

        page_text = []

        for block in blocks:
            if len(block) < 5:
                continue

            block_text = normalize_extracted_text(
                block[4]
            )

            if not block_text:
                continue

            if len(block_text.split()) < 3:
                continue

            page_text.append(block_text)

        text += "\n\n".join(page_text) + "\n\n"

    pdf.close()

    return normalize_extracted_text(text)


def extract_text(file_path):

    if file_path.endswith(".txt"):
        return read_txt(file_path)

    elif file_path.endswith(".docx"):
        return read_docx(file_path)

    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)

    else:
        raise ValueError("Unsupported file type")


def extract_pdf_pages(file_path):

    pdf = fitz.open(file_path)

    pages = []

    for page_num, page in enumerate(pdf, start=1):

        blocks = page.get_text("blocks")
        cleaned_blocks = []

        for block in blocks:
            if len(block) < 5:
                continue

            block_text = normalize_extracted_text(
                block[4]
            )

            if not block_text:
                continue

            if len(block_text.split()) < 3:
                continue

            cleaned_blocks.append(block_text)

        page_text = "\n\n".join(cleaned_blocks)

        pages.append({
            "page": page_num,
            "text": normalize_extracted_text(
                page_text
            )
        })

    pdf.close()

    return pages


if __name__ == "__main__":

    file_path = "documents/sample.pdf"

    pages = extract_pdf_pages(file_path)

    print(f"Pages Extracted: {len(pages)}")