import os
from functools import partial
from pathlib import Path

import psycopg
import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from ray.data import ActorPoolStrategy

from rag.config import EFS_DIR
from rag.data_gen import extract_sections
from rag.embed import EmbedChunks
from rag.utilities import execute_bash


class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(
            "dbname=postgres user=postgres host=localhost password=postgres"
        ) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(
                    batch["text"], batch["source"], batch["embeddings"]
                ):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)",
                        (
                            text,
                            source,
                            embedding,
                        ),
                    )
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, sql_dump_fp):
    ds = ray.data.from_items(
        [{"path": path} for path in docs_dir.rglob("*.html") if not path.is_dir()]
    )
    sections_ds = ds.flat_map(extract_sections)
    chunks_ds = sections_ds.flat_map(
        partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100,
        num_gpus=1,
        compute=ActorPoolStrategy(size=1),
    )

    embedded_chunks.map_batches(
        StoreResults,
        batch_size=128,
        num_cpus=1,
        compute=ActorPoolStrategy(size=6),
    ).count()


    execute_bash(f"sudo -u postgres pg_dump -c > {sql_dump_fp}")
    print("Updated the index!")


def load_index(
    embedding_model_name, embedding_dim, chunk_size, chunk_overlap, docs_dir=None, sql_dump_fp=None
):

    execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -c "DROP TABLE document;"')
    execute_bash(f"sudo -u postgres psql -f ../migrations/vector-{embedding_dim}.sql")
    if not sql_dump_fp:
        sql_dump_fp = Path(
            EFS_DIR,
            "sql_dumps",
            f"{embedding_model_name.split('/')[-1]}_{chunk_size}_{chunk_overlap}.sql",
        )


    if sql_dump_fp.exists():  
        execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -f {sql_dump_fp}')
    else:  
        build_index(
            docs_dir=docs_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name,
            sql_dump_fp=sql_dump_fp,
        )

    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, text, source FROM document")
            chunks = cur.fetchall()
    return chunks
