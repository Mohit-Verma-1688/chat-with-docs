
import streamlit as st
import tempfile
from dotenv import load_dotenv
import os
import glob
from multiprocessing import Pool
from typing import List
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.llms import ctransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,
    PythonLoader,
)
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
from htmlTemplates import css, user_template, bot_template

LOADER_MAPPING = {
    "csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "enex": (EverNoteLoader, {}),
   # ".eml": (MyElmLoader, {}),
    "epub": (UnstructuredEPubLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "odt": (UnstructuredODTLoader, {}),
    "pdf": (PyPDFLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    "py": (PythonLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 1500
chunk_overlap = 150
model = os.environ.get("MODEL", "mistral")

def load_model():
    callbacks=[StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)
    
    #llm = ctransformers(
        # model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    #    model = ,
        # model_file = "mistral-7b-instruct-v0.1.Q8_0.gguf",
        #model_file = "zephyr-7b-beta.Q4_0.gguf",
        # model="TheBloke/Llama-2-70B-chat-GGUF",
        # model = "Deci/DeciLM-6b-instruct",
     #    callbacks=[StreamingStdOutCallbackHandler()]
        # model_type=model_type,
        # max_new_tokens=max_new_tokens,  # type: ignore
        # temperature=temperature,  # type: ignore
    #)
    return llm

def create_vector_database(loaded_documents):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  texts = text_splitter.split_documents(loaded_documents)
  #embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
  embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
  db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
  #db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
  collection = db.get()
  db.persist()
  return db

def set_custom_prompt_condense():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONDENSE_QUESTION_PROMPT

def set_custom_prompt():
    """
    Prompt template for retrieval for each vectorstore
    """
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    #QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    #prompt = QA_CHAIN_PROMPT(context="Only answer from the provided document", question="What is the purpose of life?", name="AIBot", answer="The purpose of life is...")

    return prompt

def create_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.
    This function initializes a ConversationalRetrievalChain object with a specific chain type and configurations,
    and returns this  chain. The retriever is set up to return the top 3 results (k=3).
    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the 
        retriever.
    Returns:
        ConversationalRetrievalChain: The initialized conversational chain.
    """
    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True,input_key='query', output_key='result')
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(search_kwargs={"k": 3}),
    #     return_source_documents=True,
    #     max_tokens_limit=256,
    #     combine_docs_chain_kwargs={"prompt": prompt},
    #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    #     memory=memory,
    # )
    chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       memory = memory,
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return chain

def create_retrieval_qa_bot(loaded_documents):
    # if not os.path.exists(persist_dir):
    #       raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
     llm = load_model()  # Assuming this function exists and works as expected
     #   callbacks=[StreamingStdOutCallbackHandler()]
     #   llm = ollama(model=model, callbacks=callbacks)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    try:
        prompt = set_custom_prompt()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to get prompt: {str(e)}")

    # try:
    #     CONDENSE_QUESTION_PROMPT = set_custom_prompt_condense()  # Assuming this function exists and works as expected
    # except Exception as e:
    #     raise Exception(f"Failed to get condense prompt: {str(e)}")

    try:
        db = create_vector_database(loaded_documents)  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to get database: {str(e)}")

    try:
        # qa = create_chain(
        #     llm=llm, prompt=prompt,CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, db=db
        # )  # Assuming this function exists and works as expected
        qa = create_chain(
            llm=llm, prompt=prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa

def retrieve_bot_answer(query, loaded_documents):
    """
    Retrieves the answer to a given query using a QA bot.
    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.
    Args:
        query (str): The question to be answered by the QA bot.
    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot(loaded_documents)
    # bot_response = qa_bot_instance({"question": query})
    bot_response = qa_bot_instance({"query": query})
    # Check if the 'answer' key exists in the bot_response dictionary
    # if 'answer' in bot_response:
    #     # answer = bot_response['answer']
    #     return bot_response
    # else:
    #     raise KeyError("Expected 'answer' key in bot_response, but it was not found.")
    # result = bot_response['answer']
    result = bot_response['result']
    sources = []
    for source in bot_response["source_documents"]:
        sources.append(source.metadata['source'])
    return result

def handle_userinput(query):
   response = st.session_state.conversation({"query": query })
   #st.write(response)
   st.session_state.chat_history = response['chat_history']

   for i,message in enumerate(st.session_state.chat_history):
      if i % 2 ==0:
         st.write(user_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
      else:
         st.write(bot_template.replace(
              "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
 load_dotenv() 
 st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books")
 st.write(css, unsafe_allow_html=True)

 if "conversation"  not in st.session_state:
    st.session_state.conversation = None
 if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

 st.header("Chat with documents :books:")
 query = st.text_input("Ask a question about your Documents:")
 if query: 
    handle_userinput(query)
 #st.text_input("Ask a question about your document")

 #st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
 #st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

 with st.sidebar:
   st.subheader("your documents")
   uploaded_files = st.file_uploader("Upload your documents", 
                                     type=["pdf", "md", "txt"], 
                                     accept_multiple_files=True)
   
  
   
   if st.button("Process"):
     with st.spinner("Processing"):
      # get the pdf text 
       loaded_documents = []
       if uploaded_files :
        with tempfile.TemporaryDirectory() as td:
          # Process uploaded files
          for uploaded_file in uploaded_files:
           st.write(f"Uploaded: {uploaded_file.name}")
           ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
           st.write(f"Uploaded: {ext}")

           if ext in LOADER_MAPPING:
              # Save the uploaded file to a temporary directory
             loader_class, loader_args = LOADER_MAPPING[ext]

             file_path = os.path.join(td, uploaded_file.name)

             with open(file_path, 'wb') as temp_file:
              temp_file.write(uploaded_file.read())

           # Now create the loader instance with the file path
             loader = loader_class( file_path, **loader_args)
             loaded_documents.extend(loader.load())


             st.session_state.conversation = create_retrieval_qa_bot(loaded_documents)

             #st.write(st.session_state.conversation)
             

           else:
             st.warning(f"Unsupported file extension: {ext}")


           
           #st.session_state.converstaion = qa_bot_instance
           
          
     

if __name__ == '__main__' :
    main()