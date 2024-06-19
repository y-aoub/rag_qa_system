from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class Embedder:
    """
    A class to handle the initialization and configuration of an embedding model from HuggingFace
    """
    def __init__(self, embedding_device, hf_embedding_model_path):
        self.hf_embedding_model_path = hf_embedding_model_path
        self.embedding_device = embedding_device
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_embedding_function(self):
        """
        Initializes and return the HuggingFaceEmbeddings corresponding to an embedding function object with specified model name, device, and configuration parameters
        """
        model_kwargs = {'device': self.embedding_device, 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(model_name=self.hf_embedding_model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.logger.info(f"Embedding model {self.hf_embedding_model_path} from HuggingFace initialized with model_kwargs: {model_kwargs}, encode_kwargs: {encode_kwargs}")
        return embeddings
        
if __name__ == "__main__":
    pass

    