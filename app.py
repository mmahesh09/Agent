import os
import io
import time
import hashlib
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pypdf import PdfReader
from pyvis.network import Network

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


BASE_DIR = os.path.dirname(__file__)
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

KB_ROOT = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KB_ROOT, exist_ok=True)

st.set_page_config(
    page_title="PDF Q&A Agent",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Agent")
st.write("Upload a PDF, create a knowledge base, view it as a PyVis graph, and ask questions from it.")

if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
    st.error("API key is missing. Please add it to your .env file.")
    st.stop()

OCR_AVAILABLE = True

try:
    import fitz
    from PIL import Image
    import pytesseract
except Exception:
    OCR_AVAILABLE = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None

if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

if "page_count" not in st.session_state:
    st.session_state.page_count = None

if "kb_nodes" not in st.session_state:
    st.session_state.kb_nodes = []


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def get_kb_folder(pdf_hash: str) -> str:
    return os.path.join(KB_ROOT, pdf_hash)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def extract_pdf_text_with_pypdf(file_bytes: bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "page": page_num
                    }
                )
            )

    return documents, len(reader.pages)


def extract_pdf_text_with_ocr(file_bytes: bytes):
    documents = []
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_index + 1
                    }
                )
            )

    return documents


def prepare_nodes(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    nodes = splitter.split_documents(documents)

    for index, node in enumerate(nodes, start=1):
        node.metadata["node_id"] = f"Node-{index}"
        node.metadata["chunk_size"] = len(node.page_content)

    return nodes


def create_vector_store(nodes):
    embeddings = load_embeddings()

    return FAISS.from_documents(
        documents=nodes,
        embedding=embeddings
    )


def save_vector_store(vector_store, kb_folder):
    os.makedirs(kb_folder, exist_ok=True)
    vector_store.save_local(kb_folder)


def load_vector_store(kb_folder):
    embeddings = load_embeddings()

    return FAISS.load_local(
        kb_folder,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_llm(model_name):
    if OPENROUTER_API_KEY:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=OPENAI_API_KEY
    )


def get_available_models():
    if OPENROUTER_API_KEY:
        return {
            "GPT 4o Mini": "openai/gpt-4o-mini",
            "Gemini Flash": "google/gemini-2.0-flash-001",
            "Claude Haiku": "anthropic/claude-3.5-haiku",
            "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct",
            "Mistral 7B": "mistralai/mistral-7b-instruct"
        }

    return {
        "GPT 3.5 Turbo": "gpt-3.5-turbo"
    }


def build_context_from_docs(docs):
    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        node_id = doc.metadata.get("node_id", "Unknown Node")

        context_parts.append(
            f"[{node_id} | Page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def get_source_pages(docs):
    pages = []

    for doc in docs:
        page = doc.metadata.get("page")

        if page and page not in pages:
            pages.append(page)

    return sorted(pages)


def get_source_nodes(docs):
    nodes = []

    for doc in docs:
        node_id = doc.metadata.get("node_id")

        if node_id and node_id not in nodes:
            nodes.append(node_id)

    return nodes


def answer_question(question, docs, model_name):
    if not docs:
        return "I could not find relevant information in the PDF.", [], [], 0

    response_start_time = time.perf_counter()

    context = build_context_from_docs(docs)
    source_pages = get_source_pages(docs)
    source_nodes = get_source_nodes(docs)

    llm = get_llm(model_name)

    messages = [
        (
            "system",
            """
You are a PDF question-answering assistant.

Rules:
1. Answer only using the provided PDF context.
2. Do not use outside knowledge.
3. If the answer is not available in the PDF context, say:
   "I could not find this information in the PDF."
4. Keep the answer clear and simple.
5. Do not make up page numbers or node IDs.
"""
        ),
        (
            "user",
            f"""
PDF context:

{context}

Question:
{question}
"""
        )
    ]

    response = llm.invoke(messages)

    response_end_time = time.perf_counter()
    response_time = round(response_end_time - response_start_time, 2)

    return response.content, source_pages, source_nodes, response_time


def show_pyvis_knowledge_graph(kb_nodes, pdf_name):
    net = Network(
        height="780px",
        width="100%",
        bgcolor="#0E1117",
        font_color="white",
        directed=False,
        notebook=False,
        cdn_resources="in_line"
    )

    net.barnes_hut(
        gravity=-2600,
        central_gravity=0.25,
        spring_length=170,
        spring_strength=0.035,
        damping=0.09,
        overlap=0.4
    )

    net.add_node(
        "PDF",
        label=pdf_name,
        title=f"<b>PDF</b><br>{pdf_name}<br><br>Total nodes: {len(kb_nodes)}",
        color="#6C63FF",
        size=45,
        shape="dot"
    )

    pages_added = set()

    for node in kb_nodes:
        node_id = node.metadata.get("node_id", "Unknown Node")
        page = node.metadata.get("page", "Unknown")
        chunk_size = node.metadata.get("chunk_size", len(node.page_content))
        page_id = f"Page-{page}"
        preview = node.page_content[:700].replace("\n", " ")

        if page_id not in pages_added:
            net.add_node(
                page_id,
                label=f"Page {page}",
                title=f"<b>Page {page}</b>",
                color="#00BFA6",
                size=30,
                shape="dot"
            )

            net.add_edge(
                "PDF",
                page_id,
                color="#5555AA",
                width=2
            )

            pages_added.add(page_id)

        net.add_node(
            node_id,
            label=node_id,
            title=f"""
            <b>{node_id}</b><br>
            <b>Page:</b> {page}<br>
            <b>Characters:</b> {chunk_size}<br><br>
            {preview}
            """,
            color="#A0A0A0",
            size=18,
            shape="dot"
        )

        net.add_edge(
            page_id,
            node_id,
            color="#333366",
            width=1
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 1200
        }
      },
      "nodes": {
        "font": {
          "size": 18,
          "color": "white",
          "face": "arial"
        },
        "borderWidth": 2
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "continuous"
        }
      }
    }
    """)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        encoding="utf-8"
    ) as temp_file:
        temp_path = temp_file.name
        net.save_graph(temp_path)

    with open(temp_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    os.remove(temp_path)

    components.html(html_content, height=800, scrolling=True)


with st.sidebar:
    st.header("Settings")

    use_ocr = st.checkbox(
        "Use OCR if PDF has no readable text",
        value=False,
        help="Enable this for scanned PDFs."
    )

    if use_ocr and not OCR_AVAILABLE:
        st.warning(
            "OCR is not available. Install PyMuPDF, pillow, pytesseract, and Tesseract OCR."
        )

    st.divider()

    st.subheader("LLM Answer Mode")

    answer_mode = st.radio(
        "Choose answer mode",
        ["Single answer", "Multiple LLM answers"],
        index=0
    )

    available_models = get_available_models()

    if answer_mode == "Single answer":
        selected_model_label = st.selectbox(
            "Choose model",
            list(available_models.keys())
        )

        selected_models = {
            selected_model_label: available_models[selected_model_label]
        }

    else:
        selected_model_labels = st.multiselect(
            "Choose models to compare",
            list(available_models.keys()),
            default=list(available_models.keys())[:2]
        )

        if not selected_model_labels:
            st.warning("Select at least one model.")
            selected_model_labels = list(available_models.keys())[:1]

        selected_models = {
            label: available_models[label]
            for label in selected_model_labels
        }

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])


if uploaded_pdf:
    file_bytes = uploaded_pdf.getvalue()
    pdf_hash = get_file_hash(file_bytes)
    kb_folder = get_kb_folder(pdf_hash)

    tab_upload, tab_graph, tab_nodes, tab_chat = st.tabs(
        [
            "PDF Details",
            "Knowledge Graph",
            "Knowledge Nodes",
            "Chat"
        ]
    )

    if st.session_state.current_pdf_hash != pdf_hash:
        processing_start_time = time.perf_counter()

        with st.spinner("Reading PDF..."):
            documents, page_count = extract_pdf_text_with_pypdf(file_bytes)

        if not documents and use_ocr and OCR_AVAILABLE:
            with st.spinner("No readable text found. Running OCR..."):
                documents = extract_pdf_text_with_ocr(file_bytes)

        if not documents:
            st.error(
                "Could not extract text from this PDF. It may be scanned or image-based. Try enabling OCR from the sidebar."
            )
            st.stop()

        with st.spinner("Creating knowledge base nodes..."):
            kb_nodes = prepare_nodes(documents)

        if os.path.exists(kb_folder):
            with st.spinner("Loading existing vector knowledge base..."):
                vector_store = load_vector_store(kb_folder)

            st.session_state.vector_store = vector_store
            st.session_state.kb_nodes = kb_nodes
            st.success("PDF read successfully. Existing knowledge base loaded.")

        else:
            with st.spinner("Creating new vector knowledge base..."):
                vector_store = create_vector_store(kb_nodes)
                save_vector_store(vector_store, kb_folder)

            st.session_state.vector_store = vector_store
            st.session_state.kb_nodes = kb_nodes
            st.success("PDF read successfully. Knowledge base created.")

        processing_end_time = time.perf_counter()

        st.session_state.current_pdf_hash = pdf_hash
        st.session_state.processing_time = round(
            processing_end_time - processing_start_time,
            2
        )
        st.session_state.page_count = page_count

    vector_store = st.session_state.vector_store
    kb_nodes = st.session_state.kb_nodes

    with tab_upload:
        st.subheader("PDF Details")
        st.write(f"**File name:** {uploaded_pdf.name}")
        st.write(f"**File size:** {round(len(file_bytes) / 1024, 2)} KB")
        st.write(f"**Pages:** {st.session_state.page_count}")

        if st.session_state.processing_time is not None:
            st.write(f"**PDF processing time:** {st.session_state.processing_time} seconds")

        if kb_nodes:
            total_chars = sum(len(node.page_content) for node in kb_nodes)
            avg_chars = round(total_chars / len(kb_nodes), 2)

            unique_pages = sorted(
                list(
                    set(
                        node.metadata.get("page")
                        for node in kb_nodes
                        if node.metadata.get("page")
                    )
                )
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Nodes", len(kb_nodes))

            with col2:
                st.metric("Total Characters", total_chars)

            with col3:
                st.metric("Average Node Size", avg_chars)

            with col4:
                st.metric("Pages Indexed", len(unique_pages))

    with tab_graph:
        st.subheader("Interactive PyVis Knowledge Graph")
        st.write("Drag, zoom, hover, and explore the PDF knowledge base.")

        if kb_nodes:
            show_pyvis_knowledge_graph(kb_nodes, uploaded_pdf.name)
            st.info("Hover over a node to see its content. Use mouse wheel to zoom and drag nodes to rearrange.")
        else:
            st.info("No knowledge graph available yet.")

    with tab_nodes:
        st.subheader("Knowledge Base Nodes")

        if kb_nodes:
            search_node_text = st.text_input("Search inside nodes")

            filtered_nodes = kb_nodes

            if search_node_text:
                filtered_nodes = [
                    node for node in kb_nodes
                    if search_node_text.lower() in node.page_content.lower()
                ]

            st.write(f"Showing **{len(filtered_nodes)}** node(s)")

            for node in filtered_nodes:
                node_id = node.metadata.get("node_id", "Unknown Node")
                page = node.metadata.get("page", "Unknown")
                chunk_size = node.metadata.get(
                    "chunk_size",
                    len(node.page_content)
                )

                with st.expander(
                    f"{node_id} | Page {page} | {chunk_size} characters",
                    expanded=False
                ):
                    st.write(node.page_content)

        else:
            st.info("No knowledge base nodes available yet.")

    with tab_chat:
        st.subheader("Chat with your PDF")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message["role"] == "assistant":
                    if message.get("response_time") is not None:
                        st.caption(f"Response time: {message['response_time']} seconds")

                    if message.get("sources"):
                        st.caption(
                            f"Sources: Page(s) {', '.join(map(str, message['sources']))}"
                        )

                    if message.get("nodes"):
                        st.caption(
                            f"Source Nodes: {', '.join(message['nodes'])}"
                        )

                    if message.get("model_name"):
                        st.caption(f"Model: {message['model_name']}")

        user_question = st.chat_input("Ask a question from the PDF")

        if user_question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                total_response_start = time.perf_counter()

                with st.spinner("Searching relevant knowledge base nodes..."):
                    docs = vector_store.similarity_search(user_question, k=4)

                all_answers = []

                for model_label, model_name in selected_models.items():
                    with st.spinner(f"Generating answer with {model_label}..."):
                        answer, source_pages, source_nodes, response_time = answer_question(
                            user_question,
                            docs,
                            model_name
                        )

                    all_answers.append(
                        {
                            "model_label": model_label,
                            "model_name": model_name,
                            "answer": answer,
                            "sources": source_pages,
                            "nodes": source_nodes,
                            "response_time": response_time
                        }
                    )

                total_response_end = time.perf_counter()
                total_response_time = round(total_response_end - total_response_start, 2)

                if len(all_answers) == 1:
                    result = all_answers[0]

                    st.write(result["answer"])
                    st.caption(f"Response time: {result['response_time']} seconds")

                    if result["sources"]:
                        st.caption(
                            f"Sources: Page(s) {', '.join(map(str, result['sources']))}"
                        )

                    if result["nodes"]:
                        st.caption(
                            f"Source Nodes: {', '.join(result['nodes'])}"
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "nodes": result["nodes"],
                            "response_time": result["response_time"],
                            "model_name": result["model_label"]
                        }
                    )

                else:
                    combined_answer_for_history = ""
                    combined_nodes = []
                    combined_sources = []

                    for index, result in enumerate(all_answers, start=1):
                        with st.expander(
                            f"Answer {index}: {result['model_label']} | {result['response_time']} seconds",
                            expanded=True
                        ):
                            st.write(result["answer"])

                            if result["sources"]:
                                st.caption(
                                    f"Sources: Page(s) {', '.join(map(str, result['sources']))}"
                                )

                            if result["nodes"]:
                                st.caption(
                                    f"Source Nodes: {', '.join(result['nodes'])}"
                                )

                        combined_answer_for_history += (
                            f"Answer {index}: {result['model_label']}\n"
                            f"{result['answer']}\n\n"
                        )

                        combined_sources.extend(result["sources"])
                        combined_nodes.extend(result["nodes"])

                    combined_sources = sorted(list(set(combined_sources)))
                    combined_nodes = list(dict.fromkeys(combined_nodes))

                    st.caption(f"Total response time: {total_response_time} seconds")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": combined_answer_for_history,
                            "sources": combined_sources,
                            "nodes": combined_nodes,
                            "response_time": total_response_time,
                            "model_name": "Multiple LLMs"
                        }
                    )

else:
    st.info("Upload a PDF to begin.")