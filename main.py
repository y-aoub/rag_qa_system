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

embedder = Embedder(device='cpu', embedding_model_path=EMBEDDING_MODEL_PATH)
embdeddings = embedder.set_embeddings()

llm_client = LLMClient(model=LLM_PATH, temperature=0)
llm = llm_client.set_llm()

document_retriever = DocumentRetriever(embeddings=embdeddings, vector_store_dir_path=VECTOR_STORE_DIR_PATH, k=1)
retriever = document_retriever.set_retriever()

if __name__ == "__main__":

    data_pipeline = DataPipeline(n_files=1, embeddings=embdeddings, embeddings_device='cpu', build_vector_store=False)
    data_pipeline.run_pipeline()

    rag_chat_bot = RAGChatBot(llm=llm, retriever=retriever)  
    rag_chat_bot.get_response_loop()