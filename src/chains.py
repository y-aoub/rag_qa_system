from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.utils import DataUtils
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class RetrieverChain:
    """
    Chain that integrates a retriever with a LLM to contextualize a question
    """
    def __init__(self, retriever, llm, question_contextualizer_prompt_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retriever = retriever
        self.llm = llm
        self.question_contextualizer_prompt_path = question_contextualizer_prompt_path
        self.prompt = self.set_prompt()
        self.retriever_chain = self.set_context_retriever_chain()
        
    def set_prompt(self):
        """
        Reads and sets the prompt template from a file path
        """
        template = DataUtils.read_text(file_path=self.question_contextualizer_prompt_path)
        return PromptTemplate.from_template(template=template)
    
    def set_context_retriever_chain(self):
        """
        Creates a history-aware retriever chain using an LLM, a retriever, and a prompt
        """
        self.logger.info("RetrieverChain initialized successfuly")
        return create_history_aware_retriever(
            self.llm,
            self.retriever,
            self.prompt
            ) 

class ConversationRAGChain:
    """
    Chain that uses a retriever and LLM to handle question answering in a conversation context
    """
    def __init__(self, retriever_chain, llm, question_answerer_prompt_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retriever_chain = retriever_chain
        self.llm = llm
        self.question_answerer_prompt_path = question_answerer_prompt_path
        self.prompt = self.set_prompt()
        self.conversation_rag_chain = self.set_conversation_rag_chain()
        
        
    def set_prompt(self):
        """
        Reads and sets the prompt template from a file path
        """
        template = DataUtils.read_text(file_path=self.question_answerer_prompt_path)
        return PromptTemplate.from_template(template=template)

    def set_conversation_rag_chain(self):
        """
        Creates a retrieval chain by combining a retriever chain and a document processing chain
        """
        stuff_document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.logger.info("ConversationRAGChain initialized successfuly")
        return create_retrieval_chain(self.retriever_chain, stuff_document_chain)

    def get_response(self, chat_history, user_query):
        """
        Streams a response based on chat history and user query
        """
        response_stream = self.conversation_rag_chain.stream({
            "chat_history": chat_history,
            "input": user_query
        })
        return response_stream
    
class ChatSummarizerChain:
    """
    Chain that uses an LLM to summarize a chat conversation
    """
    def __init__(self, llm, chat_summarizer_prompt_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm = llm
        self.chat_summarizer_prompt_path = chat_summarizer_prompt_path
        self.prompt = self.set_prompt()
        self.chat_summarizer_chain = self.set_chat_summarizer_chain()

    def set_prompt(self):
        """
        Reads and sets the prompt template from the provided file path
        """
        template = DataUtils.read_text(file_path=self.chat_summarizer_prompt_path)
        return PromptTemplate.from_template(template=template)

    def set_chat_summarizer_chain(self):
        """
        Creates a chain that processes and summarizes chat text
        """
        self.logger.info("ChatSummarizerChain initialized successfuly")
        return self.prompt | self.llm | StrOutputParser()

    def summarize(self, text):
        """
        Summarizes the provided text
        """
        summary = self.chat_summarizer_chain.invoke({"text": text})
        return summary