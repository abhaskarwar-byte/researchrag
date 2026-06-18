import fitz
from docx import Document


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
        text += page.get_text()

    pdf.close()

    return text


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

        pages.append({
            "page": page_num,
            "text": page.get_text()
        })

    pdf.close()

    return pages


if __name__ == "__main__":

    file_path = "documents/sample.pdf"

    pages = extract_pdf_pages(file_path)

    print(f"Pages Extracted: {len(pages)}")