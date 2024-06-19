from src.chains import ChatSummarizerChain, RetrieverChain, ConversationRAGChain
from src.llm import LLMClient
from src.vector_store import DocumentRetriever
from src.chatbot import Chatbot

class ChatbotPipeline:
    """
    Pipeline to initialize and configure a chatbot with document retrieval chain, and a conversation summarization chain
    """
    def __init__(self, embedding_function, vector_store_dir_path,
                 n_docs, llm_path, use_ollama, huggingface_api_token, question_contextualizer_prompt_path,
                 question_answerer_prompt_path, chat_summarizer_prompt_path):

        document_retriever = DocumentRetriever(embedding_function=embedding_function,
                                            vector_store_dir_path=vector_store_dir_path,
                                            n_docs=n_docs)
        
        llm_client = LLMClient(llm_path=llm_path, temperature=0.0008, 
                               use_ollama=use_ollama, 
                               huggingface_api_token=huggingface_api_token)

        self.llm = llm_client.set_llm()
        self.retriever = document_retriever.set_retriever()
        
        self.question_contextualizer_prompt_path = question_contextualizer_prompt_path
        self.question_answerer_prompt_path = question_answerer_prompt_path
        self.chat_summarizer_prompt_path = chat_summarizer_prompt_path
    
    def init_chatbot(self):
        """
        Initializes the chatbot by creating and configuring the required chains
        """
        retriever_chain = RetrieverChain(retriever=self.retriever,
                                        llm=self.llm,
                                        question_contextualizer_prompt_path=self.question_contextualizer_prompt_path)
        
        conversation_rag_chain = ConversationRAGChain(retriever_chain=retriever_chain.retriever_chain,
                                                    llm=self.llm,
                                                    question_answerer_prompt_path=self.question_answerer_prompt_path)
        
        chat_summarizer_chain = ChatSummarizerChain(llm=self.llm,
                                                    chat_summarizer_prompt_path=self.chat_summarizer_prompt_path)
        
        return Chatbot(conversation_rag_chain=conversation_rag_chain, chat_summarizer_chain=chat_summarizer_chain)