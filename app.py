import streamlit as st
from src.data_pipeline import DataPipeline
from src.rag_chat import RAGChatBot
from src.llm import LLMClient
from src.vector_store import DocumentRetriever
from src.embedding import Embedder
from src.utils import HuggingFacePaths, LocalPaths, OllamaPaths
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LLM_PATH = OllamaPaths.LLM_PATH
EMBEDDING_MODEL_PATH = HuggingFacePaths.EMBEDDING_MODEL_PATH
VECTOR_STORE_DIR_PATH = str(LocalPaths.VECTOR_STORE_DIR_PATH)

# set embeddings
embedder = Embedder(device='cpu', embedding_model_path=EMBEDDING_MODEL_PATH)
embdeddings = embedder.set_embeddings()

# set llm client
llm_client = LLMClient(model=LLM_PATH, temperature=0)
llm = llm_client.set_llm()

# set retriever
document_retriever = DocumentRetriever(embeddings=embdeddings, vector_store_dir_path=VECTOR_STORE_DIR_PATH, k=1)
retriever = document_retriever.set_retriever()

# initialize teh RAG Chatbot
rag_chat_bot = RAGChatBot(llm=llm, retriever=retriever)

# Streamlit app
def main():
    st.title("RAG-based Chatbot (eLife and BioRxiv Articles)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in rag_chat_bot.get_response(prompt):
                full_response += response.get("answer", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

