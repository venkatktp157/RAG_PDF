from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    return " ".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )
