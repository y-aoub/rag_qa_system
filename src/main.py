from fetchers import BiorxivDataFetcher, GithubDataFetcher, HuggingFaceDataFetcher
from handlers import PDFDataHandler, XMLDataHandler, ParquetDataHandler, DocumentCreator
from embedding import Embedder, VectorStoreBuilder, DocumentRetriever
from utils.utils import LocalPaths, HuggingFacePaths
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


huggingface_paths = HuggingFacePaths()
data_path = huggingface_paths.data_path
embedding_model_path = huggingface_paths.embeddings_model_path

local_paths = LocalPaths()
vector_store_dir_path = local_paths.vector_store_dir_path


categories = ['pathology', 'neuroscience', 'immunology', 'paleontology']
n_files = 5


#################### Fetching + Processing PDFs from Biorxiv #####################

biorxiv_data_fetcher = BiorxivDataFetcher(categories=categories, start_date='2020-01-01', end_date='2024-01-01', n_files=n_files)
fetched_pdf_data = biorxiv_data_fetcher.fetch_data()

pdf_data_handler = PDFDataHandler(fetched_data=fetched_pdf_data)
processed_pdf_data = pdf_data_handler.process_fetched_data()


##################### Fetching + Processing XMLs from GitHub #####################

github_data_fetcher = GithubDataFetcher(owner='elifesciences', repo='elife-article-xml', path='articles', n_files=n_files)
fetched_xml_data = github_data_fetcher.fetch_data()

xml_data_handler = XMLDataHandler(fetched_data=fetched_xml_data)
processed_xml_data = xml_data_handler.process_fetched_data()


#################### Fetching + Processing Parquets from HuggingFace #############

huggingface_data_fetcher = HuggingFaceDataFetcher(data_path=data_path)
fetched_huggingface_data = huggingface_data_fetcher.fetch_data()

huggingface_data_handler = ParquetDataHandler(fetched_data=fetched_huggingface_data)
processed_huggingface_data = huggingface_data_handler.process_fetched_data()


#################### Making Data into Documents ####################################

document_creator = DocumentCreator(processed_pdf_data, processed_xml_data, processed_huggingface_data)
documents = document_creator.create_documents_from_data()


#################### Initializing Embedding Function ###############################

embedder = Embedder(device='cpu', embedding_model_path=embedding_model_path)
embeddings = embedder.set_embeddings()

#################### Building Vector Store #########################################

vectorstore_builder = VectorStoreBuilder(documents=documents, embeddings=embeddings, vector_store_dir_path=vector_store_dir_path)
vectorstore_builder.build_vector_store()


#################### Test Similarity Search and Vector Store Loading ################

# query = "Death evolution and explanation in the USA"
# document_retriever = DocumentRetriever(embeddings, vector_store_dir_path)
# retrieved_contents, retrieved_metadata, scores = document_retriever.mmr_similarity_search(query=query)

# for content, metadata, score in zip(retrieved_contents, retrieved_metadata, scores): 
#     print(f'→ CONTENT:--------------\n{content}\n→ METADATA:--------------\n{metadata}\n→ SCORE:--------------{score}\n-------------------------------')


# DataUtils.save_json(processed_data, file_path="processed_data.json")





