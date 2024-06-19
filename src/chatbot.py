from src.chains import ChatSummarizerChain, ConversationRAGChain
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

class Chatbot:
    """
    A class that initialize a chat session between the user and the assistant via CLI or a Streamlit app
    """
    def __init__(self, conversation_rag_chain: ConversationRAGChain,
                 chat_summarizer_chain: ChatSummarizerChain):
        self.conversation_rag_chain = conversation_rag_chain
        self.chat_summarizer_chain = chat_summarizer_chain
        self.chat_history = ""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Chatbot initialized with conversation_rag_chain, and chat_summarizer_chain")

    def get_full_response(self, user_query):
        """
        Retrieve a response to a query and streams it through a print and returns the full response
        """
        full_response = ""
        for chunk in self.conversation_rag_chain.get_response(chat_history=self.chat_history,
                                                              user_query=user_query):
            answer_chunk = chunk.get("answer", "")
            print(answer_chunk, end="", flush=True)
            full_response += answer_chunk
        return full_response

    def update_chat_history(self, user_query, full_response):
        """
        Update the chat history with the latest Human message (user) and AI Assistant massage (assistant)
        """
        self.chat_history += "\n" + "* Human: " + user_query + "\n" + "* AI Assistant: " + full_response + "\n"

    def summarize_chat_history(self):
        """
        Summarize the accumulated chat history using the summarization chain
        """
        self.chat_history = self.chat_summarizer_chain.summarize(self.chat_history).strip()

    def run_cli_chat(self):
        """
        Run a CLI chat session with the user
        """
        self.logger.info("Running chat session (CLI)")
        print("\n###########----- Chat with me! -----###########\n")
        while True:
            user_query = input("→ You: ")
            print("→ Assistant: ", end='', flush=True)
            full_response = self.get_full_response(user_query)
            print("\n")
            self.update_chat_history(user_query=user_query,
                                     full_response=full_response)
            self.summarize_chat_history()


    def handle_query(self, user_query):
        """
        Handle user queries and update the Streamlit app interface accordingly
        """
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in self.conversation_rag_chain.get_response(chat_history=self.chat_history,
                                                                  user_query=user_query):
                full_response += chunk.get("answer", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            self.update_chat_history(user_query=user_query, full_response=full_response)
            self.summarize_chat_history()

    def run_app_chat(self):
        """
        Run the Streamlit app chat interface
        """
        self.logger.info("Running chat session (Streamlit App)")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Type here...")

        # Add example questions
        with st.sidebar:
            st.markdown("### Example Questions")
            template_questions = [
                "What is insomnia?",
                "What is fMRI?",
                "How does the immune system work?",
                "What are place cells?",
                "How does the retina work?"
            ]

            for question in template_questions:
                if st.button(question):
                    user_query = question

        if user_query:
            self.handle_query(user_query)
