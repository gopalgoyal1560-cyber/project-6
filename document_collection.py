from unstructured.partition.auto import partition
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.cleaners.extract import extract_datetimetz
from unstructured.cleaners.extract import extract_email_address
from unstructured.cleaners.extract import extract_ip_address
from unstructured.cleaners.extract import extract_ip_address_name
from unstructured.cleaners.extract import extract_mapi_id
from unstructured.cleaners.extract import extract_ordered_bullets
from unstructured.cleaners.extract import extract_us_phone_number
from unstructured.chunking.basic import chunk_elements
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import re

INDIAN_PHONE_NUMBERS_RE = re.compile(r'\b(?:\+91|91|0)?[\s-]?[6-9]\d{9}\b')
# /embendings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
def extract_indian_phone_number(text: str):
    regex_match = INDIAN_PHONE_NUMBERS_RE.search(text)
    if regex_match is None:
        return ""

    start, end = regex_match.span()
    phone_number = text[start:end]
    return phone_number.strip()

def document_partition(files:list):
    if files:
        elements = []
        for file in files:
            elements.append(partition(filename=file))

        return elements

def document_cleaning(Text:str):
    text = replace_unicode_quotes(Text)
    text = clean(text,extra_whitespace=True)
    text = group_broken_paragraphs(text)
    return text

def document_extraction(Text:str):
    text = Text
    date_timetz = extract_datetimetz(text)
    email_address = extract_email_address(text)
    ip_address = extract_ip_address(text)
    ip_address_name = extract_ip_address_name(text)
    mapi_id = extract_mapi_id(text)
    ordered_bullets = extract_ordered_bullets(text)
    us_phone_number = extract_us_phone_number(text)
    indian_phone_number = extract_indian_phone_number(text)
    return [date_timetz,email_address,ip_address,ip_address_name,mapi_id,ordered_bullets,us_phone_number,indian_phone_number]

def return_chunks(elements:list):
    chunks = []
    for element in elements:
        for el in elements:
            chunks.append(chunk_elements(element))
    return chunks

def generate_embendings(chunks:list):
    embeddings = []
    for chunk in chunks:
        for ch in chunk:
            embeddings.append(model.encode(ch.text, convert_to_numpy=False))
    return embeddings
        
def layer_1(files:list):
    elements = document_partition(files)
    
    for element in elements:
        for el in element:
            el.text = document_cleaning(el.text)

    chunks = return_chunks(elements)
    embendings_list = generate_embendings(chunks)
    return elements,chunks,embendings_list

if __name__ == "__main__":
    elements,chunks,embendings_list = layer_1(["C:\\Users\\gopal\\Downloads\\Evidence_Grounded_Research_Agent_Project_Brief.html"])
    print(elements)
    print(chunks[0][0].text)
    # print(embendings_list)