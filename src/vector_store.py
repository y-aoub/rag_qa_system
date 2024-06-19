from langchain_community.vectorstores import Chroma
from pathlib import Path
import gdown
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class VectorStoreBuilder:
    """
    Class for building and a Chroma vector store from documents
    """
    def __init__(self, documents, embedding_function, vector_store_dir_path):
        self.documents = documents
        self.embedding_function = embedding_function
        self.vector_store_dir_path = vector_store_dir_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_vector_store(self):
        """
        Builds the vector store and saves it to the specified directory.
        """
        Chroma.from_documents(self.documents, self.embedding_function, persist_directory=self.vector_store_dir_path)      
        self.logger.info(f"Vectorsctore created successfully and saved to {self.vector_store_dir_path}")


class VectorStoreGdown:
    """
    Class for downloading and verifying the integrity of the vector store
    """
    def __init__(self, vector_store_dir_path, google_drive_chroma_url):
        self.vector_store_dir_path = vector_store_dir_path
        self.google_drive_chroma_url = google_drive_chroma_url
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_vector_store_integrity(self):
        """
        Checks the integrity of the vector store by verifying the presence of all the necessary files
        """
        parent_folder = Path(self.vector_store_dir_path)
        root_files = ['chroma.sqlite3']
        sub_files = ['data_level0.bin', 'header.bin', 'index_metadata.pickle', 'length.bin', 'link_lists.bin']

        if not parent_folder.is_dir():
            self.logger.warning(f"The vector store folder {parent_folder} is missing")
            return False

        for file in root_files:
            file_path = parent_folder / file
            if not file_path.is_file():
                self.logger.warning(f"The required file {file_path} is missing")
                return False

        sub_folder = None
        for item in parent_folder.iterdir():
            if item.is_dir():
                missing_files = [file for file in sub_files if not (item / file).is_file()]
                if missing_files:
                    self.logger.warning(f"The subfolder {item} is missing the following files: {', '.join(missing_files)}")
                else:
                    sub_folder = item
                    break

        if sub_folder is None:
            self.logger.warning(f"No subfolder was found in {parent_folder}")
            return False

        self.logger.info("All required files and folders of the vector store were found")
        return True
    
    def download_vector_store(self):
        """
        Downloads the vector store from the drive if the existing one is incomplete
        """
        vector_store_integrity = self.check_vector_store_integrity()
        if not vector_store_integrity:
            self.logger.info("The existing vector store is incomplete. It will be redownloaded and overwritten")
            gdown.download_folder(url=self.google_drive_chroma_url, output=self.vector_store_dir_path)
        else:
            pass
    
class DocumentRetriever:
    """
    Class for setting up a document retriever
    """
    def __init__(self, embedding_function, vector_store_dir_path, n_docs, lambda_mult= 0.5):
        self.search_kwargs = {'k': n_docs, 'fetch_k': n_docs+4, 'lambda_mult': lambda_mult}
        self.vector_store = Chroma(persist_directory=vector_store_dir_path, embedding_function=embedding_function)
        
    def set_retriever(self):
        """
        Sets the retriever with MMR as a search metric
        """
        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs=self.search_kwargs)
        return retriever