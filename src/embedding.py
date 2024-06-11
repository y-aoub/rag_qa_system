from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from typing import List

class Embedder:
    def __init__(self, device: str, embedding_model_path: str):
        self.model_name = embedding_model_path
        self.device = device

    def set_embeddings(self):
        model_kwargs = {'device': self.device, 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        return embeddings
    

class VectorStoreBuilder:
    def __init__(self, documents: List[Document], embeddings: Embeddings, vector_store_dir_path: str):
        self.documents = documents
        self.embeddings = embeddings
        self.vector_store_path = str(vector_store_dir_path)

    def build_vector_store(self):
        db = FAISS.from_documents(self.documents, self.embeddings)
        db.save_local(self.vector_store_path)
        
    
class DocumentRetriever:
    def __init__(self, embeddings: Embeddings, vector_store_dir_path: str):
        self.embeddings = embeddings
        self.vector_store_path = str(vector_store_dir_path)
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.vector_store.as_retriever
        
    def mmr_similarity_search(self, query: str, k: int = 5):
        """
        Performs MMR similarity search on the given query
        """
        embedded_query = self.embeddings.embed_query(query)
        matched_docs = self.vector_store.max_marginal_relevance_search_with_score_by_vector(embedded_query, k=k, fetch_k=k*2, lambda_mult=0.5)
        sorted_matched_docs_and_scores_by_score = sorted(matched_docs, key=lambda doc: doc[1], reverse=True)
        retrieved_contents = [doc_and_score[0].page_content for doc_and_score in sorted_matched_docs_and_scores_by_score]
        retrieved_metadata = [doc_and_score[0].metadata for doc_and_score in sorted_matched_docs_and_scores_by_score]
        scores = [doc_and_score[1] for doc_and_score in sorted_matched_docs_and_scores_by_score]
        return retrieved_contents, retrieved_metadata, scores
    
    
if __name__ == "__main__":
    pass

    