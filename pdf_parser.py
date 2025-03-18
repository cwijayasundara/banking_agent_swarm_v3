from dotenv import load_dotenv
from llama_parse import LlamaParse
from typing import List
from langchain_core.documents import Document as LangChainDocument

_ = load_dotenv()

system_prompt = """
You are provided with PDF files that contain information about bank account details
These documents contain text data and well as tables with information about the bank accounts.
You need to extract the text data and tabular data from these PDF files without loosing any information.
"""
def convert_llama_parse_docs_to_langchain_docs(llama_parse_docs: List[dict]):
    """Function to convert LlamaParse documents to Langchain documents"""

    # Convert LlamaParse documents to LangChain documents
    langchain_docs = []
    for doc in llama_parse_docs:
        # Create a LangChain Document with page_content and metadata
        langchain_doc = LangChainDocument(
            page_content=doc.get_content(),
            metadata=doc.metadata
        )
        langchain_docs.append(langchain_doc)
    return langchain_docs

def parse_pdf_file(pdf_file: str):
    """Function to parse the pdf file using LlamaParse in markdown format"""

    parser = LlamaParse(
        result_type="markdown",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="gemini-2.0-flash-001",
        invalidate_cache=True,
        system_prompt=system_prompt,
    )

    docs = parser.load_data(pdf_file)
    for doc in docs:
        doc.metadata.update({'filepath': pdf_file})

    return convert_llama_parse_docs_to_langchain_docs(docs)

def get_text_from_documents(documents):
    """Function to get the text from the parsed documents"""

    all_docs = []
    # Check if documents is already a flat list or a nested list
    if documents and not isinstance(documents[0], list):
        # If it's a flat list of documents (from parse_pdf_file)
        all_docs = documents
    else:
        # If it's a nested list of documents (from parse_pdf_files)
        for doc_list in documents:
            all_docs.extend(doc_list)

    full_text = "\n\n".join([doc.text for doc in all_docs])

    return full_text