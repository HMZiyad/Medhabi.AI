import streamlit as st
import os
from rag_chain import answer_question
from dotenv import load_dotenv

# Load env variables (API Key)
load_dotenv()

# Page config
st.set_page_config(
    page_title="Bengali Physics Tutor",
    page_icon="⚛️",
    layout="centered"
)

# Header
st.title("⚛️ পদার্থবিজ্ঞান টিউটর (Physics Tutor)")
st.markdown("""
<style>
.stChatFloatingInputContainer {
    bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
st.caption("NCTB পদার্থবিজ্ঞান ১ম পত্রের ওপর ভিত্তি করে তৈরি।")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "হ্যালো! আমি আপনার পদার্থবিজ্ঞান টিউটর। আপনার কোনো প্রশ্ন থাকলে করতে পারেন।"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন... (Ask a question)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Check for API Key
        if not os.getenv("GOOGLE_API_KEY"):
            full_response = "⚠️ **Error**: `GOOGLE_API_KEY` not found. Please check your `.env` file."
            message_placeholder.markdown(full_response)
        else:
            with st.spinner("চিন্তা করছি... (Thinking...)"):
                try:
                    # Call the RAG chain
                    response = answer_question(prompt)
                    full_response = response
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"⚠️ **Error**: {str(e)}"
                    message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
