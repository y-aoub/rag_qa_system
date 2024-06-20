from langchain_community.chat_models import ChatOllama
import subprocess
import logging
from langchain_huggingface import HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class LLMClient:
    """
    A class to initialize the LLM as a client
    """
    def __init__(self, llm_path, temperature, use_ollama, huggingface_api_token):
        self.huggingface_api_token = huggingface_api_token
        self.use_ollama = use_ollama
        self.llm_path = llm_path
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_llm_from_ollama(self):
        """
        Sets an LLM from Ollama for interacting with the LLM
        """
        llm = ChatOllama(model=self.llm_path,
                         temperature=self.temperature)
        self.logger.info(f"Using ChatOllama with model: {self.llm_path}, temperature: {self.temperature}")
        return llm

    def set_llm_from_huggingface_hub(self):
        """
        Sets HauggingFace Endpoint for interacting with the LLM
        """
        llm = HuggingFaceEndpoint(repo_id=self.llm_path,
                                  temperature=self.temperature,
                                  task="text-generation",
                                  huggingfacehub_api_token=self.huggingface_api_token,
                                  streaming=True,
                                  stop_sequences=["<|end|>"]
                                  )
        self.logger.info(f"Using HuggingFace API Endpoint with model: {self.llm_path}, temperature: {self.temperature}")
        return llm

    def set_llm(self):
        """
        Sets a client for interacting with the LLM
        """
        if self.use_ollama:
            ollama_down = OllamaDown(llm_path=self.llm_path)
            ollama_down.manage_ollama()
            return self.set_llm_from_ollama()
        else:
            return self.set_llm_from_huggingface_hub()

class OllamaDown:
    """
    Manages the installation and setup of Ollama and the required used LLM
    """
    def __init__(self, llm_path, url="https://ollama.com/install.sh"):
        self.url = url
        self.llm_path = llm_path
        self.logger = logging.getLogger(self.__class__.__name__)


    def is_ollama_installed(self):
        """
        Checks if Ollama is installed by running a version command
        """
        result = subprocess.run("ollama --version",
                                shell=True,
                                text=True,
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info(f"Ollama: {result.stdout.strip()}")
            return True
        else:
            self.logger.warning("Ollama is not installed")
            return False

    def download_ollama(self):
        """
        Downloads and installs Ollama from the Ollama download URL
        """
        result = subprocess.run(f"curl -fsSL {self.url} | sh",
                                shell=True,
                                text=True,
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info("Ollama has been installed successfully")
        else:
            self.logger.warning(f"Failed to install Ollama")

    def is_model_installed(self):
        """
        Checks if the specified LLM is already installed in Ollama
        """
        result = subprocess.run("ollama list",
                                shell=True,
                                text=True,
                                capture_output=True)
        models = result.stdout.strip().split('\n')
        if self.llm_path in models:
            self.logger.info(f"Model '{self.llm_path}' is already installed")
            return True
        else:
            return False

    def pull_ollama_model(self):
        """
        Pulls the specified LLM from Ollama if it is not already installed
        """
        result = subprocess.run(f"ollama pull {self.llm_path}",
                                shell=True,
                                text=True,
                                capture_output=True)
        if result.returncode == 0:
            self.logger.info(f"Model '{self.llm_path}' has been pulled successfully")
        else:
            self.logger.warning(f"Failed to pull model '{self.llm_path}' from Ollama models")

    def manage_ollama(self):
        """
        Manages the installation process of Ollama and the specified LLM
        """
        ollama_down = OllamaDown(model=self.llm_path)
        # check if Ollama is installed, if not, download it
        if not ollama_down.is_ollama_installed():
            ollama_down.download_ollama()

        # check if the model is installed, if not, pull it
        if not ollama_down.is_model_installed():
            ollama_down.pull_ollama_model()


if __name__ == "__main__":
    pass

