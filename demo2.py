from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
# Set the USER_AGENT environment variable
os.environ["USER_AGENT"] = "YourCustomUserAgent/1.0"
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader



links_list=['https://www.rguktrkv.ac.in/Departments.php?view=EC&staff=TS',
 'https://www.rguktrkv.ac.in/Departments.php?view=EC&staff=NTS',
 'https://www.rguktrkv.ac.in/Syllabus.php?view=ECE',
 'https://www.rguktrkv.ac.in/#']

file_paths = links_list
loader = WebBaseLoader(file_paths)
webdata = loader.load()


# Initialize the loader for the PDF file
loader = PyPDFLoader("pics&pdf/ece_syllabus_mini.pdf")
# Create a list to store the loaded pages
pages = []
# Load the pages lazily
for doc in loader.lazy_load():
    pages.append(doc)
##The lazy_load() method processes the document incrementally, loading and yielding one page at a time (or chunks of data).
##It doesn't load the entire document into memory at once. Instead, it uses an iterator to load pages one by one as needed.
##Memory Efficiency: Since it doesn't load the whole document into memory all at once, lazy_load() is more memory-efficient and is better suited for large documents.
pdf_data=pages
overalldata=pdf_data+webdata



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=50, #This sets the number of characters to overlap between consecutive chunks. Here, the text will have a 20-character overlap at the end of one chunk and the beginning of the next.
)
chunks=text_splitter.split_documents(overalldata)
print("chunks done")


cohere_api_key="txbfSbwJYRR4ogGa6dXvtz63qj5gatE0mM43LLId"
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=cohere_api_key
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(
    search_type="similarity", search_kwargs={"k":3}
)


Groq_api_key="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" # get this api key from groq website
model=ChatGroq(   # model
    temperature=0.4, 
    groq_api_key=Groq_api_key,
    model_name="llama-3.3-70b-versatile",
    max_tokens=None)
system_prompt=("You are bot specially designed for answering the queries related to ECE department in RGUKT,RK Valley"
               "if some one wishes you , greet them politely"
               "if user gives , HOD consider it as Head of the Department and also if user gives faculties consider it as all faculty in department "
               "Some times user may give improper spellings and improper setences , try to identify them and good good responses"
               "You are trained by a ece , R20 student Charan I'D number is R200037"
                "Use the data obtained from only  the retrieved context and provide the appropraite result"
               "If you dont know answer to the question say that sorry i dont know "
               "{context}"
              ) ## conetxt is autofilled
template=ChatPromptTemplate.from_messages(
    [("system",system_prompt),
    ("human","{input}"),
    ("ai","")]
)

input=input("give input : ")
question_answer_chain = create_stuff_documents_chain(model, template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain) # here 1st chunks are retrived and then it was combined with prompt to get response from llm
response=rag_chain.invoke({"input":input})
print(response["answer"])
