import requests
from typing import List, Dict
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from datasets import load_dataset, concatenate_datasets
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class BiorxivDataFetcher:
    def __init__(self, categories: List[str], start_date: str, end_date: str, n_files: int, server: str = 'biorxiv'):
        self.categories = categories
        self.start_date = start_date
        self.end_date = end_date
        self.n_files = n_files
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"BiorxivFetcher initialized with categories {self.categories}, start_date: {self.start_date}, end_date: {self.end_date}, n_files: {self.n_files}")

    def get_metadata(self, cursor: int = 0) -> Dict:
        """
        Fetches data from the biorxiv API for the given server, date range, and cursor
        """
        url = f"https://api.biorxiv.org/details/{self.server}/{self.start_date}/{self.end_date}/{cursor}/json"
        self.logger.debug(f"Fetching metadata from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def set_pdf_url(self, doi: str, version: int) -> str:
        """
        Builds the PDF URL from the DOI and version of the paper
        """
        pdf_url = f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
        self.logger.debug(f"Constructed PDF URL: {pdf_url}")
        return pdf_url
    
    def fetch_pdf_content(self, pdf_url: str) -> str:
        """
        Fetches the content (page by page) from a PDF file
        """
        loader = PyMuPDFLoader(pdf_url)
        data = loader.load()
        return data

    def fetch_data(self) -> List[Dict[str, Document]]:
        """
        Fetches metadata, PDF URLs, and text content from PDFs for the specified categories and date range, stopping when the specified number of files is reached
        """
        data = []
        fetched_files = 0
        cursor = 0

        while fetched_files < self.n_files:
            metadata = self.get_metadata(cursor)
            if 'collection' not in metadata or not metadata['collection']:
                self.logger.warning("No more collection found in the fetched data")
                break

            for paper in metadata['collection']:
                if paper['category'] in self.categories:
                    pdf_url = self.set_pdf_url(paper['doi'], paper['version'])
                    content = self.fetch_pdf_content(pdf_url)
                    self.logger.info(f"Fetched content from {pdf_url} (length: {len(content)} pages)")
                    data.append({'url': pdf_url, 'content': content})
                    fetched_files += 1
                    if fetched_files >= self.n_files:
                        break
            
            cursor += len(metadata['collection'])
        
        self.logger.info(f"Fetched content for {len(data)} files")
        return data

class GithubDataFetcher:
    def __init__(self, owner: str, repo: str, path: str, n_files: int):
        self.owner = owner
        self.repo = repo
        self.path = path
        self.n_files = n_files
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"GithubFetcher initialized with owner: {self.owner}, repo: {self.repo}, path: {self.path}")
        
    def set_api_url(self) -> str:
        """
        Builds the API URL from owner, repo and path
        """
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{self.path}"

    def get_filenames(self) -> List[Dict]:
        """
        Lists files from a GitHub repository for the given path and file extension
        """
        url = self.set_api_url()
        self.logger.debug(f"Listing files from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        files = response.json()
        filtered_files = [file for file in files if file['type'] == 'file' and file['name'].endswith(".xml")]
        self.logger.info(f"Found {len(filtered_files)}/{len(files)} XML files in the repository")
        return filtered_files

    def fetch_xml_content(self, file_url: str) -> str:
        """
        Fetches the XML content of a file from a given URL
        """
        self.logger.debug(f"Fetching file content from URL: {file_url}")
        response = requests.get(file_url)
        response.raise_for_status()
        content = response.text
        self.logger.info(f"Fetched content from {file_url} (length: {len(content)} characters)")
        return content
    
    def fetch_data(self) -> List[Dict[str, str]]:
        """
        Fetches the content of the first `n_files` files from the repository.
        """
        files = self.get_filenames()
        selected_files = files[:self.n_files]
        self.logger.debug(f"Selected files: {[file['name'] for file in selected_files]}")

        data = []

        for file in selected_files:
            xml_url = file['download_url']
            content = self.fetch_xml_content(xml_url)
            data.append({'url': xml_url, 'content': content})

        self.logger.info(f"Fetched content for {len(data)} files")
        return data
 
class HuggingFaceDataFetcher:
    def __init__(self, data_path: str):
        self.huggingface_data_path = data_path
        
    def concatenate_data(self, dataset):
        """
        Concatenate the train, test, and validation data into one dataset
        """
        concatenated_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
        return concatenated_dataset
    
    def fetch_data(self):
        """
        Load train, test, and validation data from .parquet files using HuggingFace datasets
        """
        dataset = load_dataset(path=self.huggingface_data_path, data_files={'train': 'train.parquet', 'test': 'test.parquet', 'validation': 'validation.parquet'})
        concatenated_dataset = self.concatenate_data(dataset)
        return concatenated_dataset

if __name__ == "__main__":
    pass