import streamlit as st
import requests
import json
import re

# Page configuration
st.set_page_config(
    page_title="Nirwan Raj Nagpal - Resume Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy {
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    .status-error {
        background-color: #ef4444;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .chat-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    @media (prefers-color-scheme: dark) {
        .chat-info {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
    }
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .think-content {
        display: none;
    }
    .think-toggle {
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.8rem;
        color: #666;
        cursor: pointer;
        margin: 8px 0;
        display: inline-block;
    }
    .think-toggle:hover {
        background: #e0e0e0;
    }
    @media (prefers-color-scheme: dark) {
        .think-toggle {
            background: #3a3a3a !important;
            border-color: #555 !important;
            color: #ccc !important;
        }
        .think-toggle:hover {
            background: #4a4a4a !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
BACKEND_URL = "http://localhost:8000"

def check_backend_health():
    """Check if the backend API is healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        return False, None
    except Exception as e:
        return False, str(e)

def send_chat_message(question):
    """Send a message to the chat API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question},
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        return False, "Request timed out. The model might be processing a complex query."
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def process_think_content(text):
    """Process text to handle <think></think> tags for display"""
    # Find all <think></think> blocks
    think_pattern = r'<think>(.*?)</think>'

    # Check if there are any think blocks
    think_matches = re.findall(think_pattern, text, flags=re.DOTALL)
    if not think_matches:
        return text, []

    # Remove think content from the main text
    clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text.strip())

    return clean_text, think_matches

def strip_think_content(text):
    """Strip <think></think> content from text for message history"""
    # Remove all <think></think> blocks
    think_pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL)

    # Clean up any extra whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text.strip())

    return cleaned_text

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_status" not in st.session_state:
    st.session_state.backend_status = None

# Header with status indicator
health_status, health_data = check_backend_health()
status_class = "status-healthy" if health_status else "status-error"
status_text = "Online" if health_status else "Offline"

st.markdown(f"""
<div class="main-header">
    <h1>🤖 Nirwan Raj Nagpal</h1>
    <p>Resume Assistant & Career Chatbot</p>
    <p><span class="{status_class} status-indicator"></span>Status: {status_text}</p>
</div>
""", unsafe_allow_html=True)

# Show backend status info
if health_status and health_data:
    collection_count = health_data.get('collection_count', 0)
    st.markdown(f"""
    <div class="chat-info">
        ✅ <strong>System Ready:</strong> Backend connected • {collection_count} resume sections loaded • Ollama model active
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ Backend API is not responding. Please ensure the FastAPI server is running on port 8000.")
    st.stop()

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome!** I'm Nirwan's AI resume assistant. I can help you learn about:

        - 💼 **Professional Experience** - Companies, roles, and achievements
        - 🛠️ **Technical Skills** - Programming languages, frameworks, and tools
        - 🎓 **Education & Certifications** - Academic background and training
        - 🚀 **Projects & Portfolio** - Notable work and accomplishments
        - 📍 **Career Goals** - Interests and aspirations

        Feel free to ask anything about Nirwan's background!
        """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Process think content for display (in case old messages have it)
            clean_content, think_blocks = process_think_content(message["content"])
            st.markdown(clean_content)

            # Show thinking process if it exists
            if think_blocks:
                with st.expander("🤔 Reasoning Process", expanded=False):
                    for i, think_content in enumerate(think_blocks):
                        if len(think_blocks) > 1:
                            st.write(f"**Step {i+1}:**")
                        st.text(think_content.strip())
                        if i < len(think_blocks) - 1:
                            st.write("---")
        else:
            st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources", expanded=False):
                sources = message["sources"]
                if sources:
                    st.write("Information retrieved from:")
                    for i, source in enumerate(set(sources), 1):
                        st.write(f"• {source}")
                else:
                    st.write("General knowledge response")

# Chat input
if prompt := st.chat_input("Ask about Nirwan's experience, skills, projects..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            success, response_data = send_chat_message(prompt)

            if success:
                answer = response_data.get("answer", "Sorry, I couldn't generate a response.")
                sources = response_data.get("sources", [])

                # Process and display answer with think content
                clean_content, think_blocks = process_think_content(answer)
                st.markdown(clean_content)

                # Show thinking process if it exists
                if think_blocks:
                    with st.expander("🤔 Reasoning Process", expanded=False):
                        for i, think_content in enumerate(think_blocks):
                            if len(think_blocks) > 1:
                                st.write(f"**Step {i+1}:**")
                            st.text(think_content.strip())
                            if i < len(think_blocks) - 1:
                                st.write("---")

                # Store message with think content stripped
                clean_answer = strip_think_content(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_answer,
                    "sources": sources
                })

                # Show sources
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        st.write("Information retrieved from:")
                        for i, source in enumerate(set(sources), 1):
                            st.write(f"• {source}")
            else:
                error_msg = f"❌ **Error:** {response_data}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar with additional info and controls
with st.sidebar:
    st.markdown("### 🎛️ Chat Controls")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 📊 Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")

    if health_data:
        st.markdown("### 🔧 System Status")
        st.write(f"**Backend:** {health_data.get('status', 'Unknown')}")
        st.write(f"**Ollama:** {health_data.get('ollama', 'Unknown')}")
        st.write(f"**ChromaDB:** {health_data.get('chromadb', 'Unknown')}")
        st.write(f"**Resume Sections:** {health_data.get('collection_count', 0)}")

    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What companies has Nirwan worked for?",
        "What are his main technical skills?",
        "Tell me about his education background",
        "What projects has he worked on?",
        "What programming languages does he know?"
    ]

    for question in sample_questions:
        if st.button(question, key=f"sample_{hash(question)}"):
            # Add user message and trigger processing
            st.session_state.messages.append({"role": "user", "content": question})

            # Get assistant response
            success, response_data = send_chat_message(question)

            if success:
                answer = response_data.get("answer", "Sorry, I couldn't generate a response.")
                sources = response_data.get("sources", [])

                # Store assistant message with think content stripped
                clean_answer = strip_think_content(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_answer,
                    "sources": sources
                })
            else:
                error_msg = f"❌ **Error:** {response_data}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "💬 Powered by Ollama + ChromaDB + FastAPI + Streamlit"
    "</div>",
    unsafe_allow_html=True
)