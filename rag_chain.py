import os
import argparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "l3cube-pune/bengali-sentence-similarity-sbert"
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "physics_nctb"

def get_retriever():
    """
    Initializes and returns the retriever from the local Chroma vector store.
    """
    print(f"Loading Vector Store from '{VECTOR_DB_DIR}'...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    
    # Retrieve top 4 relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    """
    Joins the content of retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """
    Constructs the RAG chain with Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, # Lower temperature for more factual answers
        google_api_key=api_key
    )

    # Prompt Template
    template = """
    You are an expert Physics teacher who speaks fluent Bengali. Your goal is to help HSC students understand physics concepts.
    Use the following pieces of retrieved context to answer the user's question.
    
    Context:
    {context}
    
    User Question: {question}
    
    Instructions:
    1. Answer strictly based on the provided Context. Do NOT use external knowledge for definitions or values.
    2. Tone: Supportive, encouraging, and clear ("বড় ভাই" or mentor persona).
    3. Language: Bengali.
    4. If the answer involves formulas, present them clearly.
    5. If the answer is not found in the context, say: "এই বিষয়টি পাঠ্যবইয়ের প্রদত্ত অংশে নেই তাই আমি উত্তর দিতে পারছি না।" (I cannot answer as this topic is not in the provided text).
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    retriever = get_retriever()
    
    # We use a chain that passes the docs through
    from langchain_core.runnables import RunnableParallel

    rag_chain_with_source = RunnableParallel(
        {"docs": retriever, "question": RunnablePassthrough()}
    ).assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
            | prompt
            | llm
            | StrOutputParser()
        )
    )
    
    return rag_chain_with_source

def answer_question(question):
    """
    Main function to answer a single question (with citations).
    """
    try:
        chain = get_rag_chain()
        print("Generating answer...")
        result = chain.invoke(question)
        
        answer_text = result["answer"]
        source_docs = result["docs"]
        
        # Format citations
        # We assume metadata has 'page'. We collect unique pages.
        unique_pages = sorted(list(set(
            doc.metadata.get("page", "Unknown") for doc in source_docs
        )))
        
        citations = ", ".join(str(p) for p in unique_pages)
        
        final_response = f"{answer_text}\n\n**Sources (Page Numbers):** {citations}"
        return final_response
        
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    # Test the chain directly if run as script
    parser = argparse.ArgumentParser(description="Ask the Bengali Physics Tutor.")
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()
    
    print(f"Question: {args.question}")
    answer = answer_question(args.question)
    print("\nAnswer:\n")
    print(answer)
