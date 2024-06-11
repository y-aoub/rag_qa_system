import os
import json
from typing import List, Dict, Any
from pathlib import Path

class DataUtils:

    @staticmethod
    def add_item_to_dict(item: Dict, key: str, value: str) -> Dict:
        item[key] = value
        return item
    
    @staticmethod
    def remove_item_from_dict(item: Dict, key:str) -> Dict:
        if key in item:
            del item[key]
        return item
    
    @staticmethod
    def merge_data(*args: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
        merged_data = []
        for dataset in args:
            merged_data.extend(dataset)
        return merged_data
            
    @staticmethod
    def save_text(text: str, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
    @staticmethod
    def split_text(text: str, sep="\n" + "="*50 + "\n") -> List:
        splitted_text = text.split(sep)
        return splitted_text
    
    @staticmethod
    def read_text(file_path: str) -> str:
        with open(file_path, 'r') as f:
            text = f.read()
        return text
            
    @staticmethod
    def save_json(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def is_file_exist(file_path: str) -> bool:
        is_file = os.path.isfile(file_path)
        return is_file
    
    @staticmethod
    def is_dir_exist(dir_path: str) -> bool:
        is_dir = os.path.isdir(dir_path)
        return is_dir
    

class LocalPaths:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir_path = Path(__file__).resolve().parent.parent
        else:
            base_dir_path = Path(base_dir).resolve()

        self.base_dir_path = base_dir_path
        
        self.data_dir_path = self.base_dir_path / 'data'
        
        self.vector_store_dir_path = self.data_dir_path / 'faiss_index'
        self.system_dir = self.data_dir_path / 'system'
        self.user_dir = self.data_dir_path / 'user'

class HuggingFacePaths:
    def __init__(self):        
        self.data_path = 'pszemraj/scientific_lay_summarisation-elife-norm'
        self.embeddings_model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.llm_model_path = 