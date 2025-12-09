"""
Streamlit frontend for Elite Craft agent.

This is the user-facing web interface that:
- Displays chat UI
- Captures user questions
- Makes HTTP requests to FastAPI backend
- Displays answers and retrieved chunks
"""
import logging
import os

import streamlit as st

from config import settings
from elite_craft.api import EliteCraftClient

os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

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
        st.markdown(message["content"])

        # Display retrieved chunks if this was an assistant message
        if message["role"] == "assistant" and "chunks" in message:
            with st.expander("📚 Retrieved Documentation", expanded=False):
                for idx, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"### Chunk {idx}")

                    # Display metadata
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(
                            f"**Source:** [{chunk['url']}]({chunk['url']})"
                        )
                    with col2:
                        similarity_pct = chunk['similarity'] * 100
                        st.markdown(
                            f"**Similarity:** "
                            f":green[{similarity_pct:.1f}%]"
                        )

                    st.markdown(
                        f"**Chunk ID:** {chunk['chunk_id_in_document']}"
                    )

                    # Display full content
                    st.markdown("**Content:**")
                    st.code(chunk['content'], language="markdown")

                    if idx < len(message["chunks"]):
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

                # Display answer
                st.markdown(response.answer)

                # Display retrieved chunks
                with st.expander("📚 Retrieved Documentation", expanded=False):
                    for idx, chunk in enumerate(
                        response.retrieved_chunks, 1
                    ):
                        st.markdown(f"### Chunk {idx}")

                        # Display metadata
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(
                                f"**Source:** [{chunk.url}]({chunk.url})"
                            )
                        with col2:
                            similarity_pct = chunk.similarity * 100
                            st.markdown(
                                f"**Similarity:** "
                                f":green[{similarity_pct:.1f}%]"
                            )

                        st.markdown(
                            f"**Chunk ID:** {chunk.chunk_id_in_document}"
                        )

                        # Display full content
                        st.markdown("**Content:**")
                        st.code(chunk.content, language="markdown")

                        if idx < len(response.retrieved_chunks):
                            st.markdown("---")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "chunks": [
                        {
                            "url": chunk.url,
                            "chunk_id_in_document": chunk.chunk_id_in_document,
                            "content": chunk.content,
                            "similarity": chunk.similarity
                        }
                        for chunk in response.retrieved_chunks
                    ]
                })

            except Exception as e:
                st.error(f"Failed to get answer: {str(e)}")
                logger.error(f"API request failed: {e}")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, LangGraph, and Claude*")