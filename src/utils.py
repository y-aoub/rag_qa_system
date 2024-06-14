import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class ArgsParser:
    """
    A class to parse command-line arguments for a data processing pipeline
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Pipeline Arguments")

        # Add arguments
        self.parser.add_argument('--embedding_device', type=str, default='cpu',
                                 choices=['cpu', 'cuda'], help="Device for embeddings (default: cpu)")
        self.parser.add_argument('--n_files', type=int, default=5,
                                 help="Number of PDFs and XMLs to extract and process (default: 5)")
        self.parser.add_argument('--build_vector_store', action='store_true',
                                 help="Flag to build Chroma vector store after data processing (default: False)")

    def parse_args(self):
        """
        Parse command-line arguments
        """
        return self.parser.parse_args()


class DataUtils:
    
    @staticmethod
    def merge_data(*args):
        merged_data = []
        for dataset in args:
            merged_data.extend(dataset)
        return merged_data
    
    @staticmethod
    def get_global_var(global_var_name):
        load_dotenv()
        return os.getenv(global_var_name)
    
class LocalPaths:
    BASE_DIR_PATH = Path(__file__).resolve().parent.parent
    DATA_DIR_PATH = BASE_DIR_PATH / 'data'
    VECTOR_STORE_DIR_PATH = DATA_DIR_PATH / 'chroma'

class HuggingFacePaths:
    DATA_PATH = 'pszemraj/scientific_lay_summarisation-elife-norm'
    EMBEDDING_MODEL_PATH = 'Alibaba-NLP/gte-large-en-v1.5'
    SUMMARIZER_MODEL_PATH = 'pszemraj/long-t5-tglobal-base-sci-simplify-elife'
    
class OllamaPaths:
    LLM_PATH = 'phi3:mini-128k'

class Prompts:
    QUESTION_CONTEXTUALIZER_SYSTEM_PROMPT = """Given a chat history and the latest user question \
        that may refer to context in the chat history, formulate a self-contained question \
            that can be understood independently, without needing the chat history.\
                Do NOT provide an answer, only rephrase the question if needed, otherwise return it as is."""

    QUESTION_ANSWERER_SYSTEM_PROMPT = """You are an assistant for question-answering tasks.\
        Use the following pieces of retrieved context to answer the question.\
            If you don't know the answer, say that you don't know.\
                Use three sentences maximum and keep the answer concise.\
                    
                    {context}"""


            