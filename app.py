import streamlit as st
import os, time, base64, pandas as pd
import plotly.express as px
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from modules.loader import load_documents
from modules.chunking import split_documents
from modules.embeddings import create_embeddings
from modules.vector_store import create_vector_store
from config import GROQ_API_KEY

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")

# ------------------------- STYLING -------------------------
st.markdown("""
<style>
/* Background gradient */
.stApp { background: linear-gradient(135deg, #020617, #0f172a); color: white; font-family: 'Segoe UI', sans-serif; }

/* Header */
.header { background: linear-gradient(90deg,#2563eb,#4f46e5); padding: 18px; border-radius: 12px; font-size: 28px; font-weight: 700; text-align:center; margin-bottom: 20px; color:white; }

/* Cards */
.card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 16px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(31,38,135,0.37); backdrop-filter: blur(8px); }

/* Metrics */
.metric { font-size: 32px; font-weight:bold; margin-top:10px; color: #60a5fa; }

/* Chat container */
.chat-container { max-height: 550px; overflow-y: auto; padding: 12px; border-radius: 16px; background: rgba(255,255,255,0.05); backdrop-filter: blur(8px); margin-bottom: 20px; }

/* Chat bubbles */
.stChatMessage_user > div:first-child { background: linear-gradient(135deg,#2563eb,#60a5fa); color:white; border-radius:20px !important; padding:12px !important; box-shadow:0 4px 10px rgba(0,0,0,0.2); }
.stChatMessage_assistant > div:first-child { background: linear-gradient(135deg,#4f46e5,#a78bfa); color:white; border-radius:20px !important; padding:12px !important; box-shadow:0 4px 10px rgba(0,0,0,0.2); }

/* Sidebar */
.sidebar .sidebar-content { background: linear-gradient(180deg,#111827,#1e293b); color:white; }

/* Input box */
.stTextInput>div>div>input { border-radius:12px; padding:10px; background-color: rgba(255,255,255,0.1); color:white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🏢 Enterprise RAG Knowledge Assistant</div>', unsafe_allow_html=True)

# ------------------------- AUTH / SESSION -------------------------
USERS = {
    "admin":{"password":"admin123","role":"admin"},
    "user1":{"password":"user123","role":"user"},
    "user2":{"password":"user234","role":"user"}
}
UPLOAD_DIR="uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role=""

if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username in USERS and USERS[username]["password"]==password:
                st.session_state.logged_in=True
                st.session_state.username=username
                st.session_state.role=USERS[username]["role"]
                st.success(f"Logged in as {username} ({st.session_state.role})")
            else:
                st.error("Invalid credentials")
    st.stop()

# ------------------------- SESSION STATE -------------------------
if "messages" not in st.session_state: st.session_state.messages=[]
if "query_count" not in st.session_state: st.session_state.query_count={}
if "response_times" not in st.session_state: st.session_state.response_times=[]

st.markdown(f"**Logged in as:** {st.session_state.username} ({st.session_state.role})")
if st.button("Logout"):
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role=""
    st.stop()

# ------------------------- DASHBOARD STATS -------------------------
cols = st.columns(4)
total_docs=len(os.listdir(UPLOAD_DIR))
total_queries=sum(st.session_state.query_count.values())
avg_time = round(sum(st.session_state.response_times)/len(st.session_state.response_times),2) if st.session_state.response_times else 0
cols[0].markdown(f'<div class="card">📄 Documents<div class="metric">{total_docs}</div></div>', unsafe_allow_html=True)
cols[1].markdown(f'<div class="card">📊 Queries<div class="metric">{total_queries}</div></div>', unsafe_allow_html=True)
cols[2].markdown(f'<div class="card">⚡ Avg Response<div class="metric">{avg_time}s</div></div>', unsafe_allow_html=True)
cols[3].markdown(f'<div class="card">🚀 Model<div class="metric">Groq</div></div>', unsafe_allow_html=True)

# ------------------------- SIDEBAR -------------------------
st.sidebar.title("📂 Document Manager")
if st.session_state.role=="admin":
    uploaded_files=st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            path=os.path.join(UPLOAD_DIR,file.name)
            with open(path,"wb") as f: f.write(file.getbuffer())
        st.sidebar.success("Upload complete")

st.sidebar.subheader("Uploaded Documents")
files=os.listdir(UPLOAD_DIR)
for file in files:
    colA,colB=st.sidebar.columns([4,1])
    with colA: st.write(file)
    with colB:
        if st.session_state.role=="admin" and st.button("❌",key=file):
            os.remove(f"{UPLOAD_DIR}/{file}")
            st.cache_resource.clear()
            st.stop()

# ------------------------- PDF VIEW -------------------------
def display_pdf(file):
    with open(file,"rb") as f: base64_pdf=base64.b64encode(f.read()).decode()
    pdf=f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px"></iframe>'
    st.markdown(pdf, unsafe_allow_html=True)

# ------------------------- QA CHAIN -------------------------
@st.cache_resource
def build_qa_chain():
    docs=[]
    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".pdf"): docs.extend(load_documents(f"{UPLOAD_DIR}/{file}"))
    if len(docs)==0: return None
    chunks=split_documents(docs)
    embeddings=create_embeddings()
    vectorstore=create_vector_store(chunks, embeddings)
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})
    llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    return RetrievalQA.from_chain_type(llm=llm,retriever=retriever,return_source_documents=True)

qa_chain=build_qa_chain()
if qa_chain is None: st.warning("Upload PDFs to enable AI chat"); st.stop()

# ------------------------- MAIN LAYOUT -------------------------
left,right=st.columns([1,2])

# LEFT - Documents
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Documents")
    files=os.listdir(UPLOAD_DIR)
    for f in files: st.write("📄",f)
    if files:
        selected=st.selectbox("Preview PDF", files)
        display_pdf(f"{UPLOAD_DIR}/{selected}")
        st.download_button("Download PDF", data=open(f"{UPLOAD_DIR}/{selected}", "rb").read(), file_name=selected)
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT - AI Chat
with right:
    st.markdown('<div class="card chat-container">', unsafe_allow_html=True)
    st.subheader("🤖 AI Chat")

    # Display chat messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    # Chat input with typing animation
    question=st.chat_input("Ask question from documents...")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        st.session_state.query_count[question]=st.session_state.query_count.get(question,0)+1

        start_time = time.time()
        with st.chat_message("assistant"):
            placeholder=st.empty()
            # Typing effect
            for i in range(3):
                placeholder.markdown("Typing" + "."*(i+1))
                time.sleep(0.3)
            response=qa_chain.invoke({"query":question})
            answer=response["result"]
            placeholder.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})
            # Show source documents
            for s in response["source_documents"]:
                st.caption(f"📄 {s.metadata.get('source','Document')}")
        st.session_state.response_times.append(round(time.time()-start_time,2))

    # Clear chat & download
    colC,colD=st.columns(2)
    with colC:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages=[]
            st.stop()
    with colD:
        if st.session_state.messages:
            chat_text="\n".join([f"{m['role'].title()}: {m['content']}" for m in st.session_state.messages])
            st.download_button("💾 Download Chat", data=chat_text, file_name="chat_history.txt")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- QUERY ANALYTICS -------------------------
st.subheader("📊 Query Analytics")
if st.session_state.query_count:
    df=pd.DataFrame(st.session_state.query_count.items(),columns=["Query","Count"]).sort_values("Count",ascending=False)
    fig=px.bar(df,x="Query",y="Count",text="Count",color="Count")
    st.plotly_chart(fig,use_container_width=True)
else:
    st.info("No analytics yet")