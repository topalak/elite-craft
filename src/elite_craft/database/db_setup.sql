  -- Enable the pgvector extension
  create extension if not exists vector;

  -- Table 1: Crawled Site's Metadata
  create table metadata (
      id serial primary key,                  -- Auto-incrementing primary key
      url varchar not null unique,       -- identifier
      source varchar not null,
      crawled_time timestamp with time zone not null,
      body_text text,   --lets keep this part just in case
  );
  -- Table 2: Chunks with Embeddings
  create table chunks (
      id serial primary key,                  -- Auto-incrementing primary key
      url varchar not null references metadata(url) on delete cascade,  -- identifier
      chunk_number integer not null,
      content text not null,
      embedding vector(768) not null,
      created_at timestamp with time zone default timezone('utc'::text, now()) not null,

      -- Prevent duplicate chunks for same article
      unique(url, chunk_number)
  );

  -- Function to search chunks with metadata
  -- Returns chunks ordered by similarity to query embedding, with optional source filtering
  create or replace function match_chunks (
    query_embedding vector(768),
    match_count int default 15,  -- todo re-rank
    source_filter varchar default null
  ) returns table (
    chunk_id bigint,
    url varchar,
    chunk_number integer,
    content text,
    source varchar,
    crawled_time timestamp with time zone,
    similarity float
  )
  language plpgsql
  as $$
  begin
    return query
    select
      c.id as chunk_id,
      c.url,
      c.chunk_number,
      c.content,
      m.source,
      m.crawled_time,
      1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    join metadata m on c.url = m.url
    where (source_filter is null or m.source = source_filter)
    order by c.embedding <=> query_embedding
    limit match_count;
  end;
  $$;

  -- Function to increment search count (optional - add search_count column if needed)
  -- CREATE OR REPLACE FUNCTION increment_search_count(target_url varchar)
  -- RETURNS void
  -- LANGUAGE plpgsql
  -- AS $$
  -- BEGIN
  --     UPDATE metadata
  --     SET search_count = search_count + 1
  --     WHERE url = target_url;
  -- END;
  -- $$;


  -- Enable RLS
  alter table metadata enable row level security;
  alter table chunks enable row level security;

  -- Public read access policies
  create policy "Allow public read access on articles"
    on metadata for select to public using (true);

  create policy "Allow public read access on chunks"
    on chunks for select to public using (true);

  -- Insert policies (adjust based on your auth needs)
  create policy "Allow public insert on articles"
    on metadata for insert to public with check (true);

  create policy "Allow public insert on chunks"
    on chunks for insert to public with check (true);


  -- UPDATE policy for search_count
  -- CREATE POLICY "Allow public update search_count on guardian_articles"
    --ON metadata FOR UPDATE TO public
    --USING (true)
    --WITH CHECK (true);




-- CRITICAL: Set timeout for authenticator role
ALTER ROLE authenticator SET statement_timeout = '2min';

-- Set timeouts for other roles
ALTER ROLE anon SET statement_timeout = '2min';
ALTER ROLE authenticated SET statement_timeout = '2min';
ALTER ROLE service_role SET statement_timeout = '5min';