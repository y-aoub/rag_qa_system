import streamlit as st
from src.utils import ParsedArgs, ArgsParser, Settings
from src.chatbot_pipeline import ChatbotPipeline
from src.data_pipeline import DataPipeline
from src.embedding import Embedder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="RAG-based Chatbot", page_icon="💻")
st.title(body="RAG-based Chatbot (eLife and BioRxiv Articles)")

@st.cache_resource 
def initialize_streamlit_app():
    # get args
    args_parser = ArgsParser()
    parsed_args = ParsedArgs(args_parser.parse_args())
    
    embedding_device = parsed_args.embedding_device
    n_files = parsed_args.n_files
    n_docs = parsed_args.n_docs
    build_vector_store = parsed_args.build_vector_store
    use_ollama = parsed_args.use_ollama
    
    # get paths and global vars
    settings = Settings()
    paths_as_strings = settings.get_paths_as_strings()
    
    llm_path = paths_as_strings["OLLAMA_LLM_PATH"] if use_ollama else paths_as_strings["HF_LLM_PATH"]
    vector_store_dir_path = paths_as_strings["VECTOR_STORE_DIR_PATH"]
    hf_embedding_model_path = paths_as_strings["HF_EMBEDDING_MODEL_PATH"]
    question_answerer_prompt_path = paths_as_strings["QUESTION_ANSWERER_PROMPT_PATH"]
    question_contextualizer_prompt_path = paths_as_strings["QUESTION_CONTEXTUALIZER_PROMPT_PATH"]
    chat_summarizer_prompt_path = paths_as_strings["CHAT_SUMMARIZER_PROMPT_PATH"]
    huggingface_api_token = paths_as_strings["HUGGINGFACE_API_TOKEN"]
    hf_summarizer_model_path = paths_as_strings["HF_SUMMARIZER_MODEL_PATH"]
    hf_data_path = paths_as_strings["HF_DATA_PATH"]
    
    # set embedding function
    embedder = Embedder(embedding_device=embedding_device, hf_embedding_model_path=hf_embedding_model_path)
    embedding_function = embedder.set_embedding_function()
    
    # run the data pipeline (fetch data -> handle data -> create vector store)
    data_pipeline = DataPipeline(n_files=n_files, embedding_function=embedding_function,
                                 hf_data_path=hf_data_path, hf_summarizer_model_path=hf_summarizer_model_path,
                                 vector_store_dir_path=vector_store_dir_path, build_vector_store=build_vector_store)
    data_pipeline.run_pipeline()
    
    # init a chatbot instantance
    chatbot_pipeline = ChatbotPipeline(embedding_function=embedding_function, vector_store_dir_path=vector_store_dir_path,
                                       n_docs=n_docs, llm_path=llm_path, use_ollama=use_ollama,
                                       huggingface_api_token=huggingface_api_token,
                                       question_contextualizer_prompt_path=question_contextualizer_prompt_path,
                                       question_answerer_prompt_path=question_answerer_prompt_path,
                                       chat_summarizer_prompt_path=chat_summarizer_prompt_path)
    chatbot = chatbot_pipeline.init_chatbot()
    return chatbot

if __name__ == "__main__":
    chatbot = initialize_streamlit_app()
    chatbot.run_app_chat()

