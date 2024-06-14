from src.llm import QuestionContextualizer, QuestionAnswerer
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class ChatHistoryManager:
    """
    A class for managing chat history for different sessions
    """
    def __init__(self):
        self.store = {}
        

    def get_session_history(self, session_id):
        """
        Gets the chat history for a given session ID (memory buffer)
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


class Chatbot:
    """
    A class for managing the overall chatbot functionality
    """
    def __init__(self, llm, retriever):
        question_contextualizer = QuestionContextualizer(llm, retriever)
        question_answerer = QuestionAnswerer(llm)
        chat_history_manager = ChatHistoryManager()
        self.history_aware_retriever = question_contextualizer.contextualize_question()
        self.rag_chain = question_answerer.answer_question(self.history_aware_retriever)
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            chat_history_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        

    def ask_question(self, query, session_id):
        """
        Asks a question to the chatbot and returns the answer
        """
        return self.conversational_rag_chain.stream({"input": query}, config={"configurable": {"session_id": session_id}})
        

class RAGChatBot:
    """
    A class for managing a chatbot based on a RAG system to answer questions
    """
    def __init__(self, llm, retriever):
        self.chatbot = Chatbot(llm, retriever)
        
    def get_conversational_rag_chain(self):
        """
        Gets the runnable from Chatbot class
        """
        return self.chatbot.conversational_rag_chain
     
    def get_response(self, query, session_id="1"):
        """
        Gets a response to a query from the chatbot.
        """
        return self.chatbot.ask_question(query=query, session_id=session_id)
    
    def stream_response(self, response):
        """
        Streams the response to a query from the chatbot
        """
        for chunk in response:
            print(chunk.get('answer', ''), end='', flush=True)
        print('\n')
        
    def get_response_loop(self, session_id="1") -> None:
        """
        Starts a loop that continuously prompts the user for input and prints the chatbot's response (used to test on the command line)
        """
        print("\n################ Start Chatting ! ################\n")
        while True:
            query = input("→ You: ")
            print("→ Assistant: ", end='', flush=True)
            self.stream_response(self.get_response(query=query, session_id=session_id))
                
            
            