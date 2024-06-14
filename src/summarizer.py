from textsum.summarize import Summarizer
from src.utils import HuggingFacePaths
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class TextSummarizer:
    def __init__(self, force_cache, token_batch_length: int = 2048):
        """
        A class used to summarize text using a pre-trained transformer model
        """
        self.summarizer = Summarizer(model_name_or_path=HuggingFacePaths.SUMMARIZER_MODEL_PATH, token_batch_length=token_batch_length, force_cache=force_cache)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"TextSummarizer initialized with model_name_or_path: {HuggingFacePaths.SUMMARIZER_MODEL_PATH}, force_cache: {force_cache}, token_batch_length: {token_batch_length}")
        
    def summarize_by_batch(self, input_text, batch_length=2048):
        """
        Summarizes the input text by splitting it into batches and summarizing each batch separately
        """
        summaries_by_batches = self.summarizer.summarize_via_tokenbatches(input_text=input_text.replace('\n', ' '), batch_length=batch_length)
        summary = ' '.join([summary_by_batch['summary'][0].replace('\n', ' ') for summary_by_batch in summaries_by_batches])
        self.logger.info(f"Content summarized (from article of {len(input_text.split(' '))} words to summary of {len(summary.split(' '))} words)")
        return summary
        
if __name__ == "__main__":
    pass
    
    
