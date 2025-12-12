--Create match_chunks function
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(768),
    match_count int DEFAULT 8,    -- todo re-rank
    similarity_threshold float DEFAULT 0.7,
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
    WITH ranked_chunks AS (
        SELECT
            c.id AS chunk_id,
            d.url,
            c.chunk_id_in_document,
            c.content,
            d.source,
            d.crawled_time,
            1 - (c.embedding <=> query_embedding) AS similarity
        FROM public.chunks c
        JOIN public.documents d ON c.document_id = d.id
        WHERE (source_filter IS NULL OR d.source = source_filter)
    )
    SELECT
        chunk_id,
        url,
        chunk_id_in_document,
        content,
        source,
        crawled_time,
        similarity
    FROM ranked_chunks
    WHERE similarity >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;   --todo check whether the function runs correctly or not
END;
$$;


-- 5) Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
