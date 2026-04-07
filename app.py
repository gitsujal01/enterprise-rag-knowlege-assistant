import streamlit as st
import os, time, base64, json
import pandas as pd
import plotly.express as px

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from modules.loader import load_documents
from modules.chunking import split_documents
from modules.embeddings import create_embeddings
from modules.vector_store import create_vector_store
from config import GROQ_API_KEY


# ------------------ CONFIG ------------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

UPLOAD_DIR = "uploads"
USER_FILE = "users.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------ USER STORAGE ------------------
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)


# ------------------ SESSION ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.messages = []
    st.session_state.query_count = {}
    st.session_state.response_times = []


# ------------------ LOGIN / SIGNUP ------------------
if not st.session_state.logged_in:

    col1, col2, col3 = st.columns([1,1.2,1])

    with col2:
        st.markdown("### 🔐 Account Access")

        tab1, tab2 = st.tabs(["Login", "Create Account"])

        users = load_users()

        # -------- LOGIN --------
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password").strip()
                login_btn = st.form_submit_button("Login")

                if login_btn:
                    user = users.get(username)
                    if user and user["password"] == password:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = user["role"]
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        # -------- SIGNUP --------
        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("New Username").strip()
                new_pass = st.text_input("New Password", type="password").strip()
                role = st.selectbox("Role", ["user", "admin"])
                signup_btn = st.form_submit_button("Create Account")

                if signup_btn:
                    if new_user in users:
                        st.warning("Username already exists")
                    elif not new_user or not new_pass:
                        st.warning("Fill all fields")
                    else:
                        users[new_user] = {
                            "password": new_pass,
                            "role": role
                        }
                        save_users(users)
                        st.success("Account created! Go to Login tab.")

    st.stop()


# ------------------ HEADER ------------------
st.title("🏢 Enterprise RAG Assistant")
st.write(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ------------------ STATS ------------------
files = os.listdir(UPLOAD_DIR)

total_docs = len(files)
total_queries = sum(st.session_state.query_count.values())
avg_time = round(sum(st.session_state.response_times)/len(st.session_state.response_times),2) if st.session_state.response_times else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", total_docs)
c2.metric("Queries", total_queries)
c3.metric("Avg Time", f"{avg_time}s")
c4.metric("Model", "Groq")


# ------------------ SIDEBAR ------------------
st.sidebar.title("📂 Documents")

if st.session_state.role == "admin":
    uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploads:
        for file in uploads:
            with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.sidebar.success("Uploaded")

st.sidebar.subheader("Files")
for f in files:
    st.sidebar.write("📄", f)


# ------------------ RAG CHAIN ------------------
@st.cache_resource
def get_chain():
    docs = []

    for f in os.listdir(UPLOAD_DIR):
        if f.endswith(".pdf"):
            docs.extend(load_documents(os.path.join(UPLOAD_DIR, f)))

    if not docs:
        return None

    chunks = split_documents(docs)
    embeddings = create_embeddings()
    store = create_vector_store(chunks, embeddings)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=store.as_retriever(k=3),
        return_source_documents=True
    )


qa = get_chain()

if not qa:
    st.warning("Upload PDFs to start chatting")
    st.stop()


# ------------------ LAYOUT ------------------
left, right = st.columns([1,2])


# -------- DOCUMENT VIEW --------
with left:
    st.subheader("📄 Documents")

    for f in files:
        st.write("📄", f)

    if files:
        selected = st.selectbox("Preview", files)

        with open(os.path.join(UPLOAD_DIR, selected), "rb") as f:
            pdf = base64.b64encode(f.read()).decode()

        st.markdown(
            f'<iframe src="data:application/pdf;base64,{pdf}" width="100%" height="400"></iframe>',
            unsafe_allow_html=True
        )


# -------- CHAT --------
with right:
    st.subheader("🤖 Chat")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.query_count = {}
            st.session_state.response_times = []
            st.rerun()

    with col2:
        if st.session_state.messages:
            chat_text = "\n".join(
                [f"{m['role'].title()}: {m['content']}" for m in st.session_state.messages]
            )
            st.download_button("💾 Download Chat", chat_text, file_name="chat.txt", mime="text/plain")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        start = time.time()

        with st.chat_message("assistant"):
            res = qa.invoke({"query": question})
            answer = res["result"]

            st.markdown(answer)

            for doc in res["source_documents"]:
                st.caption(doc.metadata.get("source", "Document"))

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.session_state.query_count[question] = st.session_state.query_count.get(question, 0) + 1
        st.session_state.response_times.append(round(time.time() - start, 2))


# ------------------ ANALYTICS ------------------
st.subheader("📊 Analytics")

if st.session_state.query_count:
    df = pd.DataFrame(
        st.session_state.query_count.items(),
        columns=["Query", "Count"]
    ).sort_values("Count", ascending=False)

    fig = px.bar(df, x="Query", y="Count", text="Count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet")