import os
import json
import uuid
import hashlib
import logging
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
logger = logging.getLogger("project-6")

def get_connection():
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set in .env")
        connection = psycopg2.connect(database_url)
        logger.info("Database connection established")
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        logger.exception(f"Error connecting to database: {e}")
        return None

def close_connection(connection):
    try:
        if connection:
            connection.close()
            logger.info("Database connection closed")
    except Exception as e:
        print(f"Error closing database connection: {e}")
        logger.exception(f"Error closing database connection: {e}")

def create_tables(connection):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename TEXT,
                    title TEXT,
                    file_hash TEXT,
                    storage_path TEXT,
                    status TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER,
                    text TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    pinecone_id TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        connection.commit()
        logger.info("Tables ready | tables=documents,chunks")
    except Exception as e:
        connection.rollback()
        print(f"Error creating tables: {e}")
        logger.exception(f"Error creating tables: {e}")

def generate_file_hash(file_path):
    try:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        logger.exception(f"Error hashing {file_path}: {e}")
        return None

def get_chunk_details(chunk):
    try:
        originals = getattr(chunk.metadata, "orig_elements", None) or [chunk]
        pages = []
        for el in originals:
            page = getattr(el.metadata, "page_number", None)
            if page is not None:
                pages.append(page)
        categories = sorted({getattr(el, "category", None) for el in originals if getattr(el, "category", None)})
        metadata = {
            "category": getattr(chunk, "category", None),
            "element_categories": categories,
            "element_count": len(originals),
            "filetype": getattr(chunk.metadata, "filetype", None),
            "languages": getattr(chunk.metadata, "languages", None)
        }
        return (min(pages) if pages else None), (max(pages) if pages else None), metadata
    except Exception as e:
        logger.exception(f"Error reading chunk details: {e}")
        return None, None, {}

def get_document_details(file_path, chunk_group):
    filename = os.path.basename(file_path)
    title = None
    author = None
    try:
        for chunk in chunk_group:
            originals = getattr(chunk.metadata, "orig_elements", None) or [chunk]
            for el in originals:
                if title is None and getattr(el, "category", None) == "Title" and getattr(el, "text", None):
                    title = el.text
            if title:
                break
    except Exception as e:
        logger.exception(f"Error reading document details: {e}")
    if not title:
        title = os.path.splitext(filename)[0]
    return filename, title

def insert_document(connection, filename, title, file_hash, storage_path, status):
    try:
        document_id = str(uuid.uuid4())
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (id, filename, title, file_hash, storage_path, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (document_id, filename, title, file_hash, storage_path, status)
            )
        connection.commit()
        logger.info(f"Document stored | document_id={document_id} | filename={filename} | status={status}")
        return document_id
    except Exception as e:
        connection.rollback()
        print(f"Error inserting document: {e}")
        logger.exception(f"Error inserting document: {e}")
        return None

def update_document_status(connection, document_id, status):
    try:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE documents SET status = %s WHERE id = %s", (status, document_id))
        connection.commit()
        logger.info(f"Document status updated | document_id={document_id} | status={status}")
    except Exception as e:
        connection.rollback()
        print(f"Error updating document status: {e}")
        logger.exception(f"Error updating document status: {e}")

def insert_chunk(connection, document_id, chunk_index, text, page_start, page_end, char_start, char_end, metadata, chunk_id=None, pinecone_id=None):
    try:
        chunk_id = chunk_id or str(uuid.uuid4())
        pinecone_id = pinecone_id or chunk_id
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chunks (id, document_id, chunk_index, text, page_start, page_end, char_start, char_end, pinecone_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (chunk_id, document_id, chunk_index, text, page_start, page_end, char_start, char_end, pinecone_id, Json(metadata or {}, dumps=lambda o: json.dumps(o, default=str)))
            )
        connection.commit()
        logger.info(f"Chunk stored | chunk_id={chunk_id} | document_id={document_id} | chunk_index={chunk_index} | chars={len(text)}")
        return chunk_id
    except Exception as e:
        connection.rollback()
        print(f"Error inserting chunk: {e}")
        logger.exception(f"Error inserting chunk: {e}")
        return None

def store_document(connection, file_path, chunk_group):
    try:
        if connection is None:
            logger.error(f"No database connection, skipping store | file={file_path}")
            return None, []
        filename, title  = get_document_details(file_path, chunk_group)
        file_hash = generate_file_hash(file_path)
        storage_path = os.path.abspath(file_path)
        document_id = insert_document(connection, filename, title, file_hash, storage_path, "processing")
        if document_id is None:
            return None, []

        chunk_ids = []
        stored = 0
        skipped = 0
        failed = 0
        offset = 0
        texts = []
        for index, chunk in enumerate(chunk_group):
            text = getattr(chunk, "text", None) or ""
            if not text.strip():
                skipped += 1
                continue
            page_start, page_end, metadata = get_chunk_details(chunk)
            char_start = offset
            char_end = offset + len(text)
            offset = char_end
            chunk_id = insert_chunk(connection, document_id, index, text, page_start, page_end, char_start, char_end, metadata)
            if chunk_id:
                chunk_ids.append(chunk_id)
                texts.append(text)
                stored += 1
            else:
                failed += 1

        logger.info(f"Store summary | document_id={document_id} | stored={stored} | skipped_empty={skipped} | failed={failed} | has_hash={file_hash is not None}")
        return document_id, chunk_ids, text
    except Exception as e:
        print(f"Error storing document {file_path}: {e}")
        logger.exception(f"Error storing document {file_path}: {e}")
        return None, []

def get_pinecone_index():
    api_key = os.getenv("PDATABASE_URL")
    index_name = os.getenv("INDEX")
    if not api_key or not index_name:
        raise ValueError("PDATABASE_URL or INDEX is not set in .env")
    index = Pinecone(api_key=api_key).Index(index_name)
    index.describe_index_stats()  
    logger.info(f"Pinecone connection established | index={index_name}")
    return index

def store_embeddings(index, chunk_ids, embeddings, batch_size=100):
    if len(chunk_ids) != len(embeddings):
        raise ValueError(f"Length mismatch | chunk_ids={len(chunk_ids)} embeddings={len(embeddings)}")
    vectors = [
        {"id": str(cid), "values": emb.tolist() if hasattr(emb, "tolist") else list(emb)}
        for cid, emb in zip(chunk_ids, embeddings)
    ]
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start:start + batch_size])
    logger.info(f"Pinecone upsert complete | vectors={len(vectors)}")
    return len(vectors)