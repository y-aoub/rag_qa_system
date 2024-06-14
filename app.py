import streamlit as st
from src.data_pipeline import DataPipeline
from src.rag_chat import RAGChatBot
from src.llm import LLMClient, OllamaDown
from src.vector_store import DocumentRetriever
from src.embedding import Embedder
from src.utils import HuggingFacePaths, LocalPaths, OllamaPaths, ArgsParser
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# parse arguments
args_parser = ArgsParser()
args = args_parser.parse_args()

embedding_device = args.embedding_device
n_files = args.n_files
build_vector_store = args.build_vector_store

# initializeuseful paths
LLM_PATH = OllamaPaths.LLM_PATH
EMBEDDING_MODEL_PATH = HuggingFacePaths.EMBEDDING_MODEL_PATH
VECTOR_STORE_DIR_PATH = str(LocalPaths.VECTOR_STORE_DIR_PATH)

# check if ollama is installed and the llm model pulled
ollama_down = OllamaDown(model=LLM_PATH)
ollama_down.manage_ollama()

# set embedding model
embedder = Embedder(device='cpu', embedding_model_path=EMBEDDING_MODEL_PATH)
embdeddings = embedder.set_embeddings()

# run the data piepleine (include: data fetching, processing and parsing)
data_pipeline = DataPipeline(n_files=n_files, embeddings=embdeddings, build_vector_store=build_vector_store)
data_pipeline.run_pipeline()

# set retriever
document_retriever = DocumentRetriever(embeddings=embdeddings, vector_store_dir_path=VECTOR_STORE_DIR_PATH, k=1)
retriever = document_retriever.set_retriever()

# set llm client
llm_client = LLMClient(model=LLM_PATH, temperature=0)
llm = llm_client.set_llm()

# initialize the RAG chatbot
rag_chat_bot = RAGChatBot(llm=llm, retriever=retriever)

# Streamlit app
def main():
    st.title("RAG-based Chatbot (eLife and BioRxiv Articles)")

    # initial chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # react to user input
    if prompt := st.chat_input("Ask me something"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # display assistant response as a chat msg
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in rag_chat_bot.get_response(prompt):
                full_response += response.get("answer", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
if __name__ == "__main__":
    # launch the streamlit app
    main()

