import os
from pathlib import Path
from dotenv import load_dotenv
import argparse
from pydantic_settings import BaseSettings
from typing import Dict

class DataUtils:
    """
    Utility class for various data operations
    """
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

    @staticmethod
    def read_text(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    

class Settings(BaseSettings):
    """
    Configuration class for setting up paths and environment global variables
    """
    BASE_DIR_PATH: Path = Path(__file__).resolve().parent.parent
    DATA_DIR_PATH: Path = BASE_DIR_PATH / 'data'
    PROMPTS_DIR_PATH: Path = DATA_DIR_PATH / "prompts"

    QUESTION_ANSWERER_PROMPT_PATH: Path = PROMPTS_DIR_PATH / "question_answerer.txt"
    QUESTION_CONTEXTUALIZER_PROMPT_PATH: Path = PROMPTS_DIR_PATH / "question_contextualizer.txt"
    CHAT_SUMMARIZER_PROMPT_PATH: Path = PROMPTS_DIR_PATH / "chat_summarizer.txt"

    VECTOR_STORE_DIR_PATH: Path = DATA_DIR_PATH / 'chroma'

    HF_DATA_PATH: str = 'pszemraj/scientific_lay_summarisation-elife-norm'
    HF_EMBEDDING_MODEL_PATH: str = 'Alibaba-NLP/gte-large-en-v1.5'
    HF_SUMMARIZER_MODEL_PATH: str = 'pszemraj/long-t5-tglobal-base-sci-simplify-elife'
    HF_LLM_PATH: str = 'microsoft/Phi-3-mini-4k-instruct'

    OLLAMA_LLM_PATH: str = 'phi3:mini-128k'

    try:
        import streamlit as st
        GOOGLE_DRIVE_CHROMA_URL: str = st.secrets["GOOGLE_DRIVE_CHROMA_URL"]
        HUGGINGFACE_API_TOKEN: str = st.secrets["HUGGINGFACE_API_TOKEN"]
    except ImportError:
        GOOGLE_DRIVE_CHROMA_URL: str = DataUtils.get_global_var("GOOGLE_DRIVE_CHROMA_URL")
        HUGGINGFACE_API_TOKEN: str = DataUtils.get_global_var("HUGGINGFACE_API_TOKEN")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

    def get_paths_as_strings(self) -> Dict[str, str]:
        """
        Returns all paths and environment as strings in a Dict structure
        """
        return {k: str(v) for k, v in self.__dict__.items()}


class ArgsParser:
    """
    A class to parse command-line arguments for a data processing pipeline
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Pipeline Arguments")
        self._add_arguments()

    def _add_arguments(self):
        """
        Add CLI arguments to the parser
        """
        self.parser.add_argument('--embedding_device', type=str, default='cpu',
                                 choices=['cpu', 'cuda'], help="Device for embeddings (default: cpu)")
        self.parser.add_argument('--n_files', type=int, default=5,
                                 help="Number of PDFs and XMLs to extract and process (default: 5)")
        self.parser.add_argument('--n_docs', type=int, default=2,
                                 help="Number of documents to retrieve through MMR similarity search (default: 1)")
        self.parser.add_argument('--build_vector_store', action='store_true',
                                 help="Flag to build Chroma vector store after fetching, processing and parsing the data (default: False)")
        self.parser.add_argument('--use_ollama', action='store_true',
                                 help="Flag to use Ollama for as LLM server (default: False)")
    
    def parse_args(self) -> argparse.Namespace:
        """
        Parse CLI arguments
        """
        return self.parser.parse_args()

class ParsedArgs:
    """
    A class to store the parsed command-line arguments
    """
    def __init__(self, args: argparse.Namespace):
        self.embedding_device: str = args.embedding_device
        self.n_files: int = args.n_files
        self.n_docs: int = args.n_docs
        self.build_vector_store: bool = args.build_vector_store
        self.use_ollama: bool = args.use_ollama