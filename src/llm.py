from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils import Prompts
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class LLMClient:
    """
    A class to initialize the LLM as a client
    """
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def set_llm(self):
        """
        Sets a client for interacting with the LLM
        """
        self.logger.info(f"Using ChatOllama with model: {self.model}, temperature: {self.temperature}")
        return ChatOllama(model=self.model, temperature=self.temperature)

class QuestionContextualizer:
    """
    A class for contextualizing questions using the LLM and a retriever
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Prompts.QUESTION_CONTEXTUALIZER_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def contextualize_question(self):
        """
        Contextualizes a question using the LLM and a retriever
        """
        return create_history_aware_retriever(self.llm, self.retriever, self.prompt)

class QuestionAnswerer:
    """
    A class for answering questions using the LLM and a history-aware retriever
    """
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Prompts.QUESTION_ANSWERER_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def answer_question(self, history_aware_retriever):
        """
        Answers a question using the LLM and a history-aware retriever
        """
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    

class OllamaDown:
    def __init__(self, model, url="https://ollama.com/install.sh"):
        self.url = url
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        

    def is_ollama_installed(self):
        result = subprocess.run("ollama --version", 
                                shell=True, 
                                text=True, 
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info(f"Ollama version: {result.stdout.strip()}")
            return True
        else:
            self.logger.warning("Ollama is not installed")
            return False

    def download_ollama(self):
        result = subprocess.run(f"curl -fsSL {self.url} | sh", 
                                shell=True, 
                                text=True, 
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info("Ollama has been installed successfully")
        else:
            self.logger.warning(f"Failed to install Ollama: {result.stderr}")

    def is_model_installed(self):
        result = subprocess.run("ollama list", 
                                shell=True, 
                                text=True, 
                                capture_output=True)
        models = result.stdout.strip().split('\n')
        if self.model in models:
            self.logger.info(f"Model '{self.model}' is already installed")
            return True
        else:
            return False

    def pull_ollama_model(self):
        result = subprocess.run(f"ollama pull {self.model}", 
                                shell=True, 
                                text=True, 
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info(f"Model '{self.model}' has been pulled successfully")
        else:
            self.logger.warning(f"Failed to pull model '{self.model}': {result.stderr}")
            
    def manage_ollama(self):
        ollama_down = OllamaDown(model=self.model)
        # check if Ollama is installed, if not, download it
        if not ollama_down.is_ollama_installed():
            ollama_down.download_ollama()

        # check if the model is installed, if not, pull it
        if not ollama_down.is_model_installed():
            ollama_down.pull_ollama_model()

    
if __name__ == "__main__":
    pass

