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
from sentence_transformers import SentenceTransformer
from collections import Counter
from datetime import datetime
import logging
import os
import platform
import re
import time
import database as db

LOG_DIR = "logfiles/document_ingestion"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"layer_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("logfiles/document_ingestion")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDIAN_PHONE_NUMBERS_RE = re.compile(r'\b(?:\+91|91|0)?[\s-]?[6-9]\d{9}\b')

logger.info(f"Session started | python={platform.python_version()} | os={platform.platform()} | log_file={os.path.abspath(LOG_FILE)}")
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
model_load_start = time.time()
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info(f"Model loaded in {time.time() - model_load_start:.2f}s | embedding_dim={model.get_sentence_embedding_dimension()} | max_seq_length={model.max_seq_length}")

def extract_indian_phone_number(text: str):
    regex_match = INDIAN_PHONE_NUMBERS_RE.search(text)
    if regex_match is None:
        return ""

    start, end = regex_match.span()
    phone_number = text[start:end]
    return phone_number.strip()

def document_partition(files:list):
    if files:
        logger.info(f"Partitioning started | file_count={len(files)}")
        elements = []
        for file in files:
            try:
                file_start = time.time()
                file_size = os.path.getsize(file)
                file_ext = os.path.splitext(file)[1].lower()
                partitioned = partition(filename=file)
                elements.append(partitioned)
                category_counts = dict(Counter(el.category for el in partitioned))
                total_chars = sum(len(el.text) for el in partitioned)
                logger.info(f"Partitioned | file={file} | extension={file_ext} | size_bytes={file_size} | elements={len(partitioned)} | total_chars={total_chars} | categories={category_counts} | time={time.time() - file_start:.2f}s")
            except FileNotFoundError as e:
                print(f"File not found: {file} - {e}")
                logger.error(f"File not found: {file} - {e}")
            except Exception as e:
                print(f"Error partitioning {file}: {e}")
                logger.exception(f"Error partitioning {file}: {e}")

        logger.info(f"Partitioning finished | files_requested={len(files)} | files_succeeded={len(elements)}")
        return elements
    logger.warning("document_partition called with empty file list")

def document_cleaning(Text:str):
    try:
        text = replace_unicode_quotes(Text)
        text = clean(text,extra_whitespace=True)
        text = group_broken_paragraphs(text)
        logger.debug(f"Cleaned text | chars_before={len(Text)} | chars_after={len(text)} | chars_removed={len(Text) - len(text)}")
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        logger.exception(f"Error cleaning text: {e}")
        return Text

def document_extraction(Text:str):
    try:
        text = Text
        date_timetz = extract_datetimetz(text)
        email_address = extract_email_address(text)
        ip_address = extract_ip_address(text)
        ip_address_name = extract_ip_address_name(text)
        mapi_id = extract_mapi_id(text)
        ordered_bullets = extract_ordered_bullets(text)
        us_phone_number = extract_us_phone_number(text)
        indian_phone_number = extract_indian_phone_number(text)
        result = [date_timetz,email_address,ip_address,ip_address_name,mapi_id,ordered_bullets,us_phone_number,indian_phone_number]
        field_names = ["datetimetz","email_address","ip_address","ip_address_name","mapi_id","ordered_bullets","us_phone_number","indian_phone_number"]
        found_fields = [name for name, value in zip(field_names, result) if value]
        logger.info(f"Extraction done | text_chars={len(text)} | fields_found={found_fields}")
        return result
    except Exception as e:
        print(f"Error extracting data: {e}")
        logger.exception(f"Error extracting data: {e}")
        return []

def return_chunks(elements:list):
    try:
        chunk = chunk_elements(elements)
        chunk_sizes = [len(ch.text) for ch in chunk]
        if chunk_sizes:
            logger.info(f"Chunking done | chunk_groups={len(chunk)} | total_chunks={len(chunk_sizes)} | min_chars={min(chunk_sizes)} | max_chars={max(chunk_sizes)} | avg_chars={sum(chunk_sizes)/len(chunk_sizes):.1f}")
        else:
            logger.warning("Chunking produced no chunks")
        return chunks
    except Exception as e:
        print(f"Error chunking elements: {e}")
        logger.exception(f"Error chunking elements: {e}")
        return chunk

def generate_embendings(chunks:list):
    embeddings = []
    try:
        embed_start = time.time()
        for chunk in chunks:
            for ch in chunk:
                embeddings.append(model.encode(ch.text, convert_to_numpy=False))
        logger.info(f"Embeddings done | embeddings={len(embeddings)} | model={EMBEDDING_MODEL_NAME} | time={time.time() - embed_start:.2f}s")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        logger.exception(f"Error generating embeddings: {e}")
        return embeddings
        
def layer_1(files:list):
    try:
        connection = db.get_connection()
        db.create_tables(connection)
        layer_start = time.time()
        logger.info(f"layer_1 started | files={files}")
        elements = document_partition(files)
        
        total_before = 0
        total_after = 0
        for element in elements:
            for el in element:
                before = len(el.text)
                el.text = document_cleaning(el.text)
                total_before += before
                total_after += len(el.text)

        logger.info(f"Cleaning summary | chars_before={total_before} | chars_after={total_after} | chars_removed={total_before - total_after} | reduction={((total_before - total_after) / total_before * 100) if total_before else 0:.2f}%")
        for e,f in zip(elements,files):
            chunks = return_chunks(elements)
            db.store_document(connection,f,chunks)

        embendings_list = generate_embendings(chunks)
        logger.info(f"layer_1 finished | documents={len(elements)} | chunk_groups={len(chunks)} | embeddings={len(embendings_list)} | total_time={time.time() - layer_start:.2f}s")
        db.close_connection(connection)
        return elements,chunks,embendings_list
    except Exception as e:
        print(f"Error in layer_1: {e}")
        logger.exception(f"Error in layer_1: {e}")
        return [],[],[]

if __name__ == "__main__":
    try:
        elements,chunks,embendings_list = layer_1(["C:\\Users\\gopal\\Downloads\\Evidence_Grounded_Research_Agent_Project_Brief.html"])
    except Exception as e:
        print(f"Error running pipeline: {e}")
        logger.exception(f"Error running pipeline: {e}")