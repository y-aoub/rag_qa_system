from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import logging

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.utils import Prompts

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger("langchain.tracers.core").setLevel(logging.ERROR)

class LLMClient:
    """
    A class to initialize the LLM as a client
    """
    def __init__(self, model: str, temperature: float):
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


    
if __name__ == "__main__":
    pass

