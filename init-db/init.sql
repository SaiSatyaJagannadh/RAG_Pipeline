CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    langchain_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content text,                                       
    embedding vector(1536),                                
    langchain_metadata json,
    category text                     
);