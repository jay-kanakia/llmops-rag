import hashlib
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.app.clients import llm,logger,app_config,ROOT_DIR

PROCESSED_TRANSCRIPT_DIR = ROOT_DIR/"data"/"processed"
VECTOR_STORE_DIR = ROOT_DIR/"saved_embeddings"

chunk_size = app_config.chunk_size
chunk_overlap = app_config.chunk_overlap

logger.info(f"Chunk Size: {chunk_size}")
logger.info(f"Chunk Overlap: {chunk_overlap}")
logger.info(f"Embedding Dimensions: {app_config.embedding_dimensions}")

embedder = OpenAIEmbeddings(model=app_config.embedder_model,
                            dimensions=app_config.embedder_dimensions)

vs  = Chroma(
    collection_name=app_config.collection_name,
    persist_directory=VECTOR_STORE_DIR.as_posix(),
    embedding_function=embedder
)

def get_retriever():
    # create the retriever
    retriever = vs.as_retriever(search_type = app_config.search_type,
                                search_kwargs = {'k':app_config.k})
    return retriever

def chunk_hash(content:str)->str:
    hashlib.sha3_256(content.encode('utf-8')).hexdigest()[:16]

# load and update the database
def upsert_document(chunk_size:int,chunk_overlap:int):
    
    # chunk the text, sized in tokens (o200k_base matches the gpt-5 model family)
    # to mirror DeepEval's ContextConstructionConfig chunk sizing
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=app_config.tokenizer_encoding,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
    )

    file_paths = sorted(PROCESSED_TRANSCRIPT_DIR.glob('*.txt'))

    skipped_files: list[str] = []
    to_embed_chunks: list[Document] = []
    to_embed_ids: list[str] = []

    for file_path in file_paths:
        try:
            text = file_path.read_text(encoding='utf-8')
        except OSError as e:
            logger.error(f"File {file_path} could not be read : {e}")
            skipped_files.append(file_path.name)
            continue

        # if no error
        doc= Document(page_content=text,metadata={'source':str(file_path)})
        doc_chunks = splitter.split_documents([doc])

        if not doc_chunks:
            skipped_files.append(file_path.name)
            continue

        candidate_ids = [f"{file_path.name}::{i}" for i in range(len(doc_chunks))]
        existing = vs.get(ids=candidate_ids, include=["metadatas"])
        existing_hash_by_id = {
            id_: (meta or {}).get("chunk_hash") for id_, meta in zip(existing["ids"], existing["metadatas"])
        }

        for chunk_id, chunk in zip(candidate_ids, doc_chunks):
            new_hash = chunk_hash(chunk.page_content)
            if existing_hash_by_id.get(chunk_id) == new_hash:
                continue
            chunk.metadata["chunk_hash"] = new_hash
            to_embed_chunks.append(chunk)
            to_embed_ids.append(chunk_id)

    if to_embed_chunks:
        vs.add_documents(to_embed_chunks, ids=to_embed_ids)

    logger.info(
        f"Transcript upsert: scanned {len(file_paths)} files, "
        f"embedded {len(to_embed_chunks)} new/changed chunks, "
        f"{len(skipped_files)} files unreadable/empty"
    )

    return {
        "files_scanned": len(file_paths),
        "files_ingested": len(file_paths) - len(skipped_files),
        "chunks_added": len(to_embed_chunks),
        "skipped_files": skipped_files,
    }