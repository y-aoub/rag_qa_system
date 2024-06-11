from textsum.summarize import Summarizer
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class TextSummarizer:
    def __init__(self, force_cache, model_name_or_path: str = "pszemraj/long-t5-tglobal-base-sci-simplify-elife", token_batch_length: int = 2048):
        self.summarizer = Summarizer(model_name_or_path=model_name_or_path, token_batch_length=token_batch_length, force_cache=force_cache)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"TextSummarizer initialized with model_name_or_path: {model_name_or_path}, force_cache: {force_cache}, token_batch_length: {token_batch_length}")
        
    def summarize_by_batch(self, input_text: str, batch_length: int = 2048) -> str:
        summaries_by_batches = self.summarizer.summarize_via_tokenbatches(input_text=input_text.replace('\n', ' '), batch_length=batch_length)
        summary = ' '.join([summary_by_batch['summary'][0].replace('\n', ' ') for summary_by_batch in summaries_by_batches])
        self.logger.info(f"Content summarized (from article of {len(input_text.split(' '))} words to summary of {len(summary.split(' '))} words)")
        return summary
        
if __name__ == "__main__":
    pass
    
    
