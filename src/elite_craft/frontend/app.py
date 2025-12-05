"""
Streamlit frontend for Elite Craft agent.

This is the user-facing web interface that:
- Displays chat UI
- Captures user questions
- Makes HTTP requests to FastAPI backend
- Displays answers and retrieved chunks
"""
import logging

import requests
from requests.exceptions import RequestException
import streamlit as st

from config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

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
                    response = requests.post(
                        f"{API_BASE_URL}/api/update-db",
                        json={"urls": urls},
                        timeout=10
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {data['message']}")
                        st.info(
                            "Processing in background. "
                            "Check API logs for progress."
                        )
                    else:
                        st.error(
                            f"Failed to start update: "
                            f"{response.status_code}"
                        )

                except RequestException as e:
                    st.error(f"Failed to connect to API: {str(e)}")
                    logger.error(f"Database update request failed: {e}")

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
            with st.expander("📚 Retrieved Documentation"):
                for idx, chunk in enumerate(message["chunks"], 1):
                    st.markdown(
                        f"**Source {idx}:** "
                        f"[{chunk['url']}]({chunk['url']})"
                    )
                    st.markdown(
                        f"*Chunk ID: {chunk['chunk_id_in_document']}*"
                    )
                    st.code(
                        chunk['content'][:300] + "...",
                        language="markdown"
                    )
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
                response = requests.post(
                    f"{API_BASE_URL}/api/ask",
                    json={"query": query},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display answer
                    st.markdown(data["answer"])

                    # Display retrieved chunks
                    with st.expander("📚 Retrieved Documentation"):
                        for idx, chunk in enumerate(
                            data["retrieved_chunks"], 1
                        ):
                            st.markdown(
                                f"**Source {idx}:** "
                                f"[{chunk['url']}]({chunk['url']})"
                            )
                            st.markdown(
                                f"*Chunk ID: "
                                f"{chunk['chunk_id_in_document']}*"
                            )
                            st.code(
                                chunk['content'][:300] + "...",
                                language="markdown"
                            )
                            st.markdown("---")

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "chunks": data["retrieved_chunks"]
                    })
                else:
                    st.error(f"API Error: {response.status_code}")
                    logger.error(
                        f"API returned status code: "
                        f"{response.status_code}"
                    )

            except RequestException as e:
                st.error(f"Failed to connect to API: {str(e)}")
                logger.error(f"API request failed: {e}")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain, LangGraph, and Claude*")