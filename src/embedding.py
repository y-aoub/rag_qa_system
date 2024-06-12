from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from typing import List
import logging


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class Embedder:
    def __init__(self, device: str, embedding_model_path: str):
        self.model_name = embedding_model_path
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_embeddings(self):
        model_kwargs = {'device': self.device, 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.logger.info(f"Embedding model {self.model_name} initialized with model_kwargs: {model_kwargs}, encode_kwargs: {encode_kwargs}")
        return embeddings
    

# class VectorStoreBuilder:
#     def __init__(self, documents: List[Document], embeddings: Embeddings, vector_store_dir_path: str):
#         self.documents = documents
#         self.embeddings = embeddings
#         self.vector_store_path = str(vector_store_dir_path)

#     def build_vector_store(self):
#         db = FAISS.from_documents(self.documents, self.embeddings)
#         db.save_local(self.vector_store_path)
        
class VectorStoreBuilder:
    def __init__(self, documents: List[Document], embeddings: Embeddings, vector_store_dir_path: str):
        self.documents = documents
        self.embeddings = embeddings
        self.vector_store_path = str(vector_store_dir_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_vector_store(self):
        self.logger.info(f"Creating vectorstore ...")
        Chroma.from_documents(self.documents, self.embeddings, persist_directory=self.vector_store_path)      
        self.logger.info(f"Vectorsctore created successfully and saved to {self.vector_store_path}")
        
    
class MMRDocumentRetriever:
    def __init__(self, embeddings: Embeddings, vector_store_dir_path: str, k: int, lambda_mult: int= 0.5):
        self.search_kwargs = {'k': k, 'fetch_k': 2*k, 'lambda_mult': lambda_mult}
        vector_store = FAISS.load_local(str(vector_store_dir_path), embeddings, allow_dangerous_deserialization=True)
        self.retriever = vector_store.as_retriever(search_type="mmr")
        
    def retrieve_document(self, query: str):
        docs = self.retriever.invoke(query)
        return docs
    
    
if __name__ == "__main__":
    pass

    