
import logging
from src.fetchers import BiorxivDataFetcher, GithubDataFetcher, HuggingFaceDataFetcher
from src.handlers import PDFDataHandler, XMLDataHandler, ParquetDataHandler, DocumentCreator
from src.summarizer import TextSummarizer
from src.vector_store import VectorStoreBuilder, VectorStoreGdown
from src.utils import LocalPaths, HuggingFacePaths

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


class DataPipeline:
    """
    A class to handle the end-to-end data processing pipeline, including data fetching,
    processing, and optionally building or downloading a vector store
    """
    def __init__(self, n_files, embeddings, embeddings_device: str, build_vector_store: bool = False):
        self.n_files = n_files
        self.embeddings = embeddings
        self.embeddings_device = embeddings_device
        self.build_vector_store = build_vector_store
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DataPipeline initialized with n_files: {self.n_files}, embeddings: {self.embeddings}, embedding_device: {self.embeddings_device}, build_vector_store: {self.build_vector_store}")
    
    def run_data_fetchers(self):
        """
        Runs data fetchers to retrieve data from various sources
        """
        biorxiv_data_fetcher = BiorxivDataFetcher(categories=['pathology', 'neuroscience', 'paleontology'], start_date='2020-01-01', end_date='2024-01-01', n_files=self.n_files)
        fetched_pdf_data = biorxiv_data_fetcher.fetch_data()
        
        github_data_fetcher = GithubDataFetcher(owner='elifesciences', repo='elife-article-xml', path='articles', n_files=self.n_files)
        fetched_xml_data = github_data_fetcher.fetch_data()
        
        huggingface_data_fetcher = HuggingFaceDataFetcher(data_path=HuggingFacePaths.DATA_PATH)
        fetched_huggingface_data = huggingface_data_fetcher.fetch_data()
        
        return fetched_pdf_data, fetched_xml_data, fetched_huggingface_data
    

    def run_data_handlers(self, fetched_pdf_data, fetched_xml_data, fetched_huggingface_data):
        """
        Processes fetched data using appropriate handlers and summarizers
        """
        summarizer = TextSummarizer(force_cache=False)
        
        pdf_data_handler = PDFDataHandler(summarizer=summarizer, fetched_data=fetched_pdf_data)
        processed_pdf_data = pdf_data_handler.process_fetched_data()

        xml_data_handler = XMLDataHandler(summarizer=summarizer, fetched_data=fetched_xml_data)
        processed_xml_data = xml_data_handler.process_fetched_data()

        huggingface_data_handler = ParquetDataHandler(fetched_data=fetched_huggingface_data)
        processed_huggingface_data = huggingface_data_handler.process_fetched_data()

        document_creator = DocumentCreator(processed_pdf_data, processed_xml_data, processed_huggingface_data)
        documents = document_creator.create_documents_from_data()
        
        return documents
    
    def run_pipeline(self):
        """
        Runs the complete data pipeline, either building a new vector store or downloading it from the drive
        """
        if self.build_vector_store:
            
            fetched_pdf_data, fetched_xml_data, fetched_huggingface_data = self.run_data_fetchers()
            documents = self.run_data_handlers(fetched_pdf_data, fetched_xml_data, fetched_huggingface_data)
            
            vectorstore_builder = VectorStoreBuilder(documents=documents, embeddings=self.embeddings, vector_store_dir_path=LocalPaths.VECTOR_STORE_DIR_PATH)
            vectorstore_builder.build_vector_store()

        else:
            vector_store_gdown = VectorStoreGdown(vector_store_dir_path=LocalPaths.VECTOR_STORE_DIR_PATH)
            vector_store_gdown.download_vector_store()
        
if __name__ == "__main__":
    pass