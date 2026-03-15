import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

def process_pdf(pdf_file, chunk_size=2000, chunk_overlap=200):
    """
    Reads a PDF file and splits it into manageable chunks for translation.
    """
    # Streamlit UploadedFile needs to be written to a temp file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue()) # Use getvalue() for uploaded file
        tmp_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
