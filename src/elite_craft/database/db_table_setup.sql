-- 2) Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 3) Create documents table in private schema
CREATE TABLE documents (
    id serial primary key,          -- Auto-incrementing primary key
    url varchar not null unique,    -- identifier
    source varchar not null,
    crawled_time timestamp with time zone not null,
    body_preview text
);

-- 4) Create chunks table in private schema
CREATE TABLE chunks (
    id serial primary key,          -- Auto-incrementing primary key
    document_id integer not null,
    chunk_id_in_document integer not null,
    content text not null,
    embedding vector(768) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,

    -- Prevent duplicate chunks for same document
    unique(document_id, chunk_id_in_document),

    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- 5) Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);

-- 6) Set timeouts
ALTER ROLE authenticator SET statement_timeout = '2min';
ALTER ROLE anon SET statement_timeout = '2min';
ALTER ROLE authenticated SET statement_timeout = '2min';
ALTER ROLE service_role SET statement_timeout = '5min';

-- Enable RLS (optional, service_role bypasses it anyway)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;