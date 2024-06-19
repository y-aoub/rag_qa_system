import xml.etree.ElementTree as ET
from src.utils import DataUtils
from langchain.docstore.document import Document
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class XMLDataHandler:
    """
    A class to handle and process fetched XML data by parsing and summarizing content
    """
    def __init__(self, summarizer, fetched_data):
        self.fetched_data = fetched_data
        self.summarizer = summarizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"XMLDataHandler initialized with fetched_data (length: {len(fetched_data)})")
        
    def get_paragraphs_from_xml(self, xml_content):
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
    
    def get_title_from_xml(self, xml_content):
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

    
    def get_year_from_xml(self, xml_content):
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

    def process_fetched_data(self):
        """
        Processes the fetched data by extracting paragraphs from sections
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
    """
    A class to handle and process fetched PDF data by parsing and summarizing content
    """
    def __init__(self, summarizer, fetched_data):
        self.fetched_data = fetched_data
        self.summarizer = summarizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"PDFDataHandler initialized with fetched_data (length {len(fetched_data)})")
    
    def get_title_from_pdf(self, pdf_content):
        """
        Extracts the title from a Document object.
        """
        if 'title' in pdf_content.metadata:
            self.logger.info(f"Title found: {pdf_content.metadata['title']}")
            return pdf_content.metadata['title']
        else:
            self.logger.warning("Title not found")
            return ""
    
    def get_year_from_pdf(self, pdf_content):
        """
        Extracts the year from a Document object
        """
        if 'creationDate' in pdf_content.metadata:
            self.logger.info(f"Year found: {pdf_content.metadata['creationDate'][2:6]}")
            return pdf_content.metadata['creationDate'][2:6]
        else:
            self.logger.warning("Year not found")
            return ""
    
    def get_paragraphs_from_pdf(self, pdf_content):
        return ''.join([page.page_content for page in pdf_content])
        
    def process_fetched_data(self):
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
    """
    A class to handle and process data fetched from Parquet files
    """
    def __init__(self, fetched_data):
        self.fetched_data = fetched_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"HuggingFaceDataHandler initialized with fetched_data (length {len(fetched_data)})")
        
    def filter_huggingface_data(self, keys_to_keep):
        """
        Filters fetched data to keep specified keys
        """
        self.logger.info(f"Data filtered by keeping keys: {keys_to_keep}")
        return [{k: d[k] for k in keys_to_keep} for d in self.fetched_data]
    
    def process_fetched_data(self):
        """
        Processes fetched data by filtering and adding data source information
        """
        source = "https://huggingface.co/datasets/pszemraj/scientific_lay_summarisation-elife-norm" 
        filtered_data = self.filter_huggingface_data(keys_to_keep = ['summary', 'year', 'title'])
        for item in filtered_data:
            item['source'] = source
        self.logger.info(f"{source} added as data source to the processed data")
        return filtered_data

class DocumentCreator:
    """
    A class to create transform processed data items into documents 
    """
    def __init__(self, *args):
        self.processed_data = DataUtils.merge_data(*args)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_document_from_dict(self, item):
        """
        Creates a Document object from a dictionary item
        """
        page_content = item['summary']
        metadata = {'publication_year': item['year'], 'article_source': item['source'], 'article_title': item['title']}
        return Document(page_content=page_content, metadata=metadata)

    def create_documents_from_data(self):
        """
        Creates Document objects from processed data
        """
        self.logger.info(f"Creating documents for {len(self.processed_data)} items with 'summary' as page content and 'article_title', 'publication_year', 'article_source' as metadata")
        return list(map(self.create_document_from_dict, self.processed_data))


if __name__ == "__main__":
    pass