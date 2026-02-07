CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id serial primary key,          -- Auto-incrementing primary key
    url varchar not null unique,    -- identifier
    source varchar not null,
    crawled_time timestamp with time zone not null,
    body_preview text
);

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


ALTER ROLE authenticator SET statement_timeout = '2min';
ALTER ROLE anon SET statement_timeout = '2min';
ALTER ROLE authenticated SET statement_timeout = '2min';
ALTER ROLE service_role SET statement_timeout = '5min';

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;