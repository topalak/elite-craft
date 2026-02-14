"""
Streamlit frontend for Elite Craft agent.

This is the user-facing web interface that:
- Displays chat UI
- Captures user questions
- Makes HTTP requests to Django backend
- Displays answers
"""
import logging
import os

import streamlit as st

from config import settings
from elite_craft.api import EliteCraftClient

os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'true')
os.environ['LANGSMITH_ENDPOINT'] = getattr(settings, 'LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY.get_secret_value()
os.environ['LANGSMITH_PROJECT'] = getattr(settings, 'LANGSMITH_PROJECT', 'elite-craft')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = EliteCraftClient(host=settings.API_HOST, port=settings.API_PORT)

# Page configuration
st.set_page_config(
    page_title="Elite Craft - AI Agent Helper",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Elite Craft")
st.markdown(
    "*Your AI assistant for building agents with "
    "LangChain & LangGraph*"
)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Elite Craft helps developers build agentic AI projects "
        "by providing expert knowledge on LangChain, LangGraph, "
        "and more."
    )

    st.markdown("---")
    st.markdown("### Database Management")

    # Text area for URLs
    urls_input = st.text_area(
        "URLs to add (one per line)",
        placeholder=(
            "https://docs.langchain.com/...\n"
            "https://docs.langchain.com/..."
        ),
        height=150
    )

    # Update database button
    if st.button(
        "Update Database",
        type="primary",
        use_container_width=True
    ):
        if not urls_input.strip():
            st.warning("Please enter at least one URL")
        else:
            # Parse URLs from text area
            # Split by newlines, filter empty lines
            urls = [
                url.strip()
                for url in urls_input.split("\n")
                if url.strip()
            ]

            with st.spinner(
                f"Starting database update for {len(urls)} URLs..."
            ):
                try:
                    response = client.update_database(urls)

                    st.success(f"✅ {response.message}")
                    st.info(
                        "Processing in background. "
                        "Check API logs for progress."
                    )

                except Exception as e:
                    st.error(f"Failed to update database: {str(e)}")
                    logger.error(f"Database update failed: {e}")

# Main chat interface
st.header("Ask a Question")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show chunk count if this message has chunks
        if message.get("chunk_count"):
            st.info(f"📊 Retrieved {message['chunk_count']} chunk{'s' if message['chunk_count'] != 1 else ''} for this query")

        st.markdown(message["content"])

        # Display chunks if available
        if message.get("chunks"):
            with st.expander(
                f"📚 Retrieved Chunks ({len(message['chunks'])})",
                expanded=False
            ):
                for i, chunk in enumerate(message['chunks'], 1):
                    if isinstance(chunk, dict):
                        # Display chunk header with similarity and source
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            source = chunk.get('source', 'unknown')
                            chunk_num = chunk.get('chunk_number', '?')
                            st.markdown(f"**Chunk {i}** - `{source}` (chunk #{chunk_num})")
                        with col2:
                            similarity = chunk.get('similarity', 0)
                            st.metric("Similarity", f"{similarity:.3f}")

                        # Display URL
                        if 'url' in chunk:
                            st.caption(f"🔗 [{chunk['url']}]({chunk['url']})")

                        # Display content
                        content = chunk.get('content', '')
                        if content:
                            st.markdown(content)

                        # Display crawled time if available
                        if 'crawled_time' in chunk:
                            st.caption(f"⏰ Crawled: {chunk['crawled_time']}")

                    else:
                        # Fallback for non-dict chunks
                        st.markdown(f"**Chunk {i}**")
                        st.markdown(str(chunk))

                    st.markdown("---")

# Chat input
if query := st.chat_input("How do I build an agent?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.ask_question(query)

                # Display chunk count indicator
                chunk_count = len(response.retrieved_chunks) if response.retrieved_chunks else 0
                if chunk_count > 0:
                    st.info(f"📊 Retrieved {chunk_count} chunk{'s' if chunk_count != 1 else ''} for this query")

                # Display answer
                st.markdown(response.answer)

                # Display retrieved chunks for debugging
                if response.retrieved_chunks:
                    with st.expander(
                        f"📚 Retrieved Chunks ({len(response.retrieved_chunks)})",
                        expanded=False
                    ):
                        for i, chunk in enumerate(response.retrieved_chunks, 1):
                            if isinstance(chunk, dict):
                                # Display chunk header with similarity and source
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    source = chunk.get('source', 'unknown')
                                    chunk_num = chunk.get('chunk_number', '?')
                                    st.markdown(f"**Chunk {i}** - `{source}` (chunk #{chunk_num})")
                                with col2:
                                    similarity = chunk.get('similarity', 0)
                                    st.metric("Similarity", f"{similarity:.3f}")

                                # Display URL
                                if 'url' in chunk:
                                    st.caption(f"🔗 [{chunk['url']}]({chunk['url']})")

                                # Display content
                                content = chunk.get('content', '')
                                if content:
                                    # Render as markdown to preserve code blocks
                                    st.markdown(content)

                                # Display crawled time if available
                                if 'crawled_time' in chunk:
                                    st.caption(f"⏰ Crawled: {chunk['crawled_time']}")

                            else:
                                # Fallback for non-dict chunks
                                st.markdown(f"**Chunk {i}**")
                                st.markdown(str(chunk))

                            st.markdown("---")

                # Add to history with chunks
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "chunks": response.retrieved_chunks,
                    "chunk_count": chunk_count
                })

            except Exception as e:
                st.error(f"Failed to get answer: {str(e)}")
                logger.error(f"API request failed: {e}")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, LangGraph, and Claude*")