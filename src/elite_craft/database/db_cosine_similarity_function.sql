--Create match_chunks function
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(768),
    match_count int DEFAULT 20,    -- todo re-rank
    source_filter varchar DEFAULT NULL
) RETURNS TABLE (
    chunk_id integer,
    url varchar,
    chunk_id_in_document integer,
    content text,
    source varchar,
    crawled_time timestamp with time zone,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        d.url,                                -- get url from documents, not chunks
        c.chunk_id_in_document,
        c.content,
        d.source,
        d.crawled_time,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id  -- correct JOIN condition
    WHERE (source_filter IS NULL OR d.source = source_filter)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- 5) Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
