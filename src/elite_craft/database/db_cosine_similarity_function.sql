--Create match_chunks function
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(768),
    match_count int DEFAULT 5,    -- todo re-rank
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
