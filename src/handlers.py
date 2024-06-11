import sys
sys.path.append('..')

import xml.etree.ElementTree as ET
from typing import List, Dict
from langchain.docstore.document import Document
from summarizer import TextSummarizer
from utils.utils import DataUtils
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class XMLDataHandler:
    def __init__(self, fetched_data: List[Dict[str, List[str]]]):
        self.fetched_data = fetched_data
        self.summarizer = TextSummarizer(force_cache=False)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"XMLDataHandler initialized with fetched_data (length: {len(fetched_data)})")
        
    def get_paragraphs_from_xml(self, xml_content: str) -> str:
        """
        Extracts all paragraphs from the sec components in the given XML content
        """
        paragraphs = []
        root = ET.fromstring(xml_content)
        for sec in root.findall('.//sec'):
            for p in sec.findall('.//p'):
                paragraph_text = ''.join(p.itertext())
                if paragraph_text.strip():
                    paragraphs.append(paragraph_text.strip())
        self.logger.info(f"Paragraphs text content extracted (length: {len(paragraphs)} paragraphs)")
        return ' '.join(paragraphs)
    
    def get_title_from_xml(self, xml_content: str) -> str:
        """
        Extracts the title from the given XML content
        """
        root = ET.fromstring(xml_content)
        title_element = root.find('.//article-title')
        if title_element is not None:
            title = ''.join(title_element.itertext()).strip()
            self.logger.info(f"Title found: {title}")
            return title
        else:
            self.logger.warning("Title not found")
            return ""

    
    def get_year_from_xml(self, xml_content: str) -> str:
        """
        Extracts the copyright year as the publication year from the given XML content
        """
        root = ET.fromstring(xml_content)
        year_element = root.find('.//copyright-year')
        if year_element is not None:
            self.logger.info(f"Year found: {year_element.text.strip()}")
            return year_element.text.strip()
        else:
            self.logger.warning("Year not found")
            return ""

    def process_fetched_data(self) -> List[Dict[str, str]]:
        """
        Processes the fetched data by extracting paragraphs from sections.
        Returns a list of dictionaries where each dictionary contains a key 'url' and the extracted content.
        """
        processed_data = []

        for item in self.fetched_data:
            source = item['url']
            xml_content = item['content']
            content = self.get_paragraphs_from_xml(xml_content)
            title = self.get_title_from_xml(xml_content)
            year = self.get_year_from_xml(xml_content)
            processed_data.append({'summary': self.summarizer.summarize_by_batch(content), 'year': year, 'title': title, 'source' : source})
        self.logger.info(f"{source} added as data source to the processed data")
        return processed_data
    
class PDFDataHandler:
    def __init__(self, fetched_data: List[Dict[str, List[Document]]]):
        self.fetched_data = fetched_data
        self.summarizer = TextSummarizer(force_cache=False)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"PDFDataHandler initialized with fetched_data (length {len(fetched_data)})")
    
    def get_title_from_pdf(self, pdf_content: Document) -> str:
        """
        Extracts the title from a Document object.
        Returns the title as a string.
        """
        if 'title' in pdf_content.metadata:
            self.logger.info(f"Title found: {pdf_content.metadata['title']}")
            return pdf_content.metadata['title']
        else:
            self.logger.warning("Title not found")
            return ""
    
    def get_year_from_pdf(self, pdf_content: Document) -> str:
        """
        Extracts the year from a Document object.
        """
        if 'creationDate' in pdf_content.metadata:
            self.logger.info(f"Year found: {pdf_content.metadata['creationDate'][2:6]}")
            return pdf_content.metadata['creationDate'][2:6]
        else:
            self.logger.warning("Year not found")
            return ""
    
    def get_paragraphs_from_pdf(self, pdf_content: List[Document]) -> str:
        return ''.join([page.page_content for page in pdf_content])
        
    def process_fetched_data(self) -> List[Dict[str, str]]:
        """
        Processes the fetched data and returns a list of dictionaries where each dictionary contains keys 'summary', 'year', 'title' and 'source'
        """
        processed_data = []
        
        for item in self.fetched_data:
            source = item['url']
            content = self.get_paragraphs_from_pdf(item['content'])
            title = self.get_title_from_pdf(item['content'][0])
            year = self.get_year_from_pdf(item['content'][0])
            processed_data.append({'summary': self.summarizer.summarize_by_batch(content), 'year': year, 'title': title, 'source' : source})
        self.logger.info(f"{source} added as data source to the processed data")
        return processed_data
    
class ParquetDataHandler:
    def __init__(self, fetched_data):
        self.fetched_data = fetched_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"HuggingFaceDataHandler initialized with fetched_data (length {len(fetched_data)})")
        
    def filter_huggingface_data(self, keys_to_keep: List[str]):
        self.logger.info(f"Data filtered by keeping keys: {keys_to_keep}")
        return [{k: d[k] for k in keys_to_keep} for d in self.fetched_data]
    
    def process_fetched_data(self):
        source = "https://huggingface.co/datasets/pszemraj/scientific_lay_summarisation-elife-norm" 
        filtered_data = self.filter_huggingface_data(keys_to_keep = ['summary', 'year', 'title'])
        for item in filtered_data:
            item['source'] = source
        self.logger.info(f"{source} added as data source to the processed data")
        return filtered_data

class DocumentCreator:
    def __init__(self, *args):
        self.processed_data = DataUtils.merge_data(*args)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_document_from_dict(self, item: Dict[str, str]) -> Document:
        page_content = item['summary']
        metadata = {'publication_year': item['year'], 'article_source': item['source'], 'article_title': item['title']}
        doc = Document(page_content=page_content, metadata=metadata)
        return doc

    def create_documents_from_data(self) -> List[Document]:
        self.logger.info(f"Creating documents for {len(self.processed_data)} items with 'summary' as page content and 'article_title', 'publication_year', 'article_source' as metadata")
        return list(map(self.create_document_from_dict, self.processed_data))


if __name__ == "__main__":
    pass