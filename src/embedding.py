from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class Embedder:
    """
    A class to handle the initialization and configuration of an embedding model from HuggingFace
    """
    def __init__(self, device, embedding_model_path):
        self.model_name = embedding_model_path
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_embeddings(self):
        """
        Initialize and return the HuggingFaceEmbeddings object with specified model name, device, and configuration parameters
        """
        model_kwargs = {'device': self.device, 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.logger.info(f"Embedding model {self.model_name} initialized with model_kwargs: {model_kwargs}, encode_kwargs: {encode_kwargs}")
        return embeddings
        
if __name__ == "__main__":
    pass

    