from src.data_pipeline import DataPipeline
from src.rag_chat import RAGChatBot
from src.llm import LLMClient
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


if __name__ == "__main__":

    # set embedding model
    embedder = Embedder(device=embedding_device, embedding_model_path=EMBEDDING_MODEL_PATH)
    embdeddings = embedder.set_embeddings()

    # set llm client
    data_pipeline = DataPipeline(n_files=n_files, embeddings=embdeddings, build_vector_store=build_vector_store)
    data_pipeline.run_pipeline()
    
    # set retriever
    document_retriever = DocumentRetriever(embeddings=embdeddings, vector_store_dir_path=VECTOR_STORE_DIR_PATH, k=1)
    retriever = document_retriever.set_retriever()

    llm_client = LLMClient(model=LLM_PATH, temperature=0)
    llm = llm_client.set_llm()

    # launch a command-line chat (a test version, but can be useful for quick use)
    rag_chat_bot = RAGChatBot(llm=llm, retriever=retriever)  
    rag_chat_bot.get_response_loop()