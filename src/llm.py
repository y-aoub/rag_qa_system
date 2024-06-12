from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from utils.utils import DataUtils

class LargeLanguageModel:
    def __init__(self, llm_path: str, use_api: bool = True, max_new_tokens: int = 512, temperature: float = 0.8):
        self.llm_path = llm_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_api = use_api
            
    def set_llm_api(self):
        huggingfacehub_api_token = DataUtils.get_global_var('HUGGINGFACEHUB_API_TOKEN')
        llm = HuggingFaceEndpoint(
        repo_id=self.llm_path,
        temperature=self.temperature,
        task="text-generation",
        max_new_tokens=self.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.03,
        timeout=240,
        huggingfacehub_api_token=huggingfacehub_api_token)
        return llm
    
    def set_llm_local(self):
        llm = HuggingFacePipeline.from_model_id(
        model_id=self.llm_path,
        task="text-generation",
        pipeline_kwargs=dict(max_new_tokens=self.max_new_tokens,
                             do_sample=False,
                             repetition_penalty=1.03,
                             temperature=self.temperature))
        return llm

    def set_llm(self):
        if self.use_api:
            return self.set_llm_api()
        else:
            return self.set_llm_local()

class ContextSummarizer:
    def __init__():
        pass
    
class QA:
    pass

class Chat:
    pass