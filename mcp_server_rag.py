# math_server.py
from mcp.server.fastmcp import FastMCP
import os
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from PyPDF2 import PdfReader

mcp = FastMCP("mcp_server")

def chunk_processing(pdf):
    """
    Process a PDF file, extracting text and splitting it into chunks.
    """
    pdf_reader = PdfReader(pdf)
        
    text = ""
    for page in pdf_reader.pages:        
        text += page.extract_text()
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def embeddings(chunks):
    """
    Create embeddings for text chunks using OpenAI.
    """
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    version = os.environ["AZURE_OPENAI_API_VERSION"]
    api_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment = os.environ["AZURE_OPENAI_MODEL_NAME_EMB"]

    # Initialize OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        openai_api_version=version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )
    # Create vector store using FAISS
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def generation(VectorStore, query:str) -> str:
    """
    Generate responses based on prompts and embeddings.
    """
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    version = os.environ["AZURE_OPENAI_API_VERSION"]
    api_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment = os.environ["AZURE_OPENAI_MODEL_NAME"]

    model = AzureChatOpenAI(
        api_key=api_key,
        api_version=version,
        azure_endpoint=api_endpoint,
        model=deployment,
        max_tokens=16000
    )

    retriever = VectorStore.as_retriever()
    
    # Define template for prompts
    template = """Respond to the prompt based on the following context: {context}
    Questions: {questions}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Define processing chain
    chain = (
        {"context": retriever, "questions": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    # Invoke the processing chain with user input
    output = chain.invoke(query)
    return output


@mcp.tool()
def get_files_list() -> dict:
    """The tool returns information of the files which is stored in this server.
    Parameter: no parameter;
    Return: return file name and file descriptions in JSON format.
    """
   
    data_directory = 'data'  
    files = {  
        "files": []  
    }

    # 遍历 data 目录中的所有文件  
    for filename in os.listdir(data_directory):  
        if filename.endswith('.pdf'):  
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            version = os.environ["AZURE_OPENAI_API_VERSION"]
            api_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            deployment = os.environ["AZURE_OPENAI_MODEL_NAME_16K"]

            text = ""  
            pdf_path = os.path.join(data_directory, filename)
            pdf = open(pdf_path, 'rb')
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(chunk_size=16000, chunk_overlap=0)  
            chunks = text_splitter.split_text(text)  
            
            # 初始化 Langchain 的 LLM  
            llm = AzureChatOpenAI(
                model=deployment,
                azure_endpoint=api_endpoint,
                api_key=api_key,
                api_version=version
                )  # 确保你有合适的 API 密钥  

            summaries = ""  
            for chunk in chunks:  
                # Instantiate chain
                message = SystemMessage(
                        content="In Chinese, write a concise summary of the following:\\n\\n"+chunk
                    )
                summary = llm.invoke([message])
                summaries = summaries + summary.content

            files["files"].append({"File Name": filename, "File Summary": summaries})

    return files

@mcp.tool()
def answer_question(question:str) -> str:
    """This tool supports users to ask questions in natural language about the documents saved in this tool.
    The tool is developed using RAG technology and stores necessary information and knowledge in a vector database.
    Users can ask questions to the tool in natural language, and the tool will answer the user's questions based on the information and knowledge stored in the tool using natural language.
    Parameter: Customer's question
    Return: Answer returned based on the customer's question
    """
    # 指定数据目录  
    data_directory = 'data'  
    all_chunks = []

    # 遍历 data 目录中的所有文件  
    for filename in os.listdir(data_directory):  
        if filename.endswith('.pdf'):  
            pdf_path = os.path.join(data_directory, filename)  
            
            # Open the PDF file in binary mode
            pdf = open(pdf_path, 'rb') 
            # Process the PDF file into chunks
            processed_chunks = chunk_processing(pdf)
            all_chunks = all_chunks + processed_chunks

    # Embed the processed chunks using OpenAI embeddings
    embedded_chunks = embeddings(all_chunks)

    # Generate responses based on the embedded chunks
    generated_response = generation(embedded_chunks, query=question)

    # Print the generated response
    print("\n in mcp server"+generated_response+"\n")

    return generated_response

if __name__ == "__main__":
    #filelist_json = get_files_list()
    answer_str = answer_question("薪酬体系的结构和组成是怎样的？")
    mcp.run(transport="stdio")