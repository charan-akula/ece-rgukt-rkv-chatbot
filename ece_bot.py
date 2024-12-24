from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st # for interface
from fpdf import FPDF # for converting the text into pdf 
import base64
#import os
# Set the USER_AGENT environment variable
#os.environ["USER_AGENT"] = "YourCustomUserAgent/1.0"
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader


# Set the page title and custom favicon (this will appear in the browser tab)
#st.set_page_config(page_title="ECE RGUKT,RKV", page_icon=":guardsman:")
# Title and logo
title = "ECE RGUKT,RKV Chat Assistant"
logo_path = "pics&pdf/images.jpg"
# Function to convert image to base64 for embedding in HTML
def get_base64_image(image_path):
    """Converts an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# Add the markdown for the title with one image
st.markdown(
    f"""
    <style>
    .title-container {{
        display: flex;
        align-items: center;
        justify-content: left;
        margin-bottom: 40px;  /* Add space below the title */
    }}
    .title {{
        font-size: 40px;
        font-weight: bold;
        color: gold;  /* Gold color for the title text */
        text-decoration: underline;
        margin-right: 20px;  /* Space between the title and the image */
        margin-top: -20px;
    }}
    .logo {{
        width: 40px;
        height: 40px;
    }}
    </style>
    <div class="title-container">
        <div class="title">{title}</div>
        <img src="data:image/png;base64,{get_base64_image(logo_path)}" class="logo" alt="Collage Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Define user and system logos (replace with your image paths or URLs)
user_logo = "pics&pdf/user.jpeg"
system_logo = "pics&pdf/bot.jpeg"
# Function to generate the chat history as a PDF
# Function to generate the chat history as a PDF
def generate_pdf():
    # Create PDF instance
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Set font for the title
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Chat History", ln=True, align="C")
    pdf.ln(10)

    # Loop through messages to add them to the PDF
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Set red color and bold font for the user heading
            pdf.set_text_color(255, 0, 0)  # Red color for headings
            pdf.set_font("Arial", "B", 12)  # Bold font for user headings
            pdf.cell(0, 10, "User:", ln=True)

            # Set black color and normal font for the user message content
            pdf.set_text_color(0, 0, 0)  # Black color for content
            pdf.set_font("Arial", "", 12)  # Normal font for user content
            pdf.multi_cell(0, 10, message["content"])

        elif message["role"] == "system":
            # Set red color and bold font for the system heading
            pdf.set_text_color(255, 0, 0)  # Red color for headings
            pdf.set_font("Arial", "B", 12)  # Bold font for system headings
            pdf.cell(0, 10, "System:", ln=True)

            # Set black color and normal font for the system message content
            pdf.set_text_color(0, 0, 0)  # Black color for content
            pdf.set_font("Arial", "", 12)  # Normal font for system content
            pdf.multi_cell(0, 10, message["content"])

        # Add a small gap between each message
        pdf.ln(5)

    # Output the PDF to a binary stream (for download)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# Initialize session state to store messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to display chat messages with avatars
def display_messages():
    for message in st.session_state.messages:
        avatar = user_logo if message["role"] == "user" else system_logo
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


# Get user input using `st.chat_input`




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
               "if user gives , HOD consider it as Head of the Department and also if user gives faculties consider it as all faculty "
               "if some one wishes you , greet them politely"
               "Some times user may give improper spellings and improper setences , try to identify them and good good responses"
               "You are trained by a ece , R20 student Charan I'D number is R200037"
                "Use the data obtained from only  the retrieved context and provide the appropraite result "
               "If you dont know answer to the question say that sorry i dont know "
               "{context}"
              ) ## conetxt is autofilled
template=ChatPromptTemplate.from_messages(
    [("system",system_prompt),
    ("human","{input}"),
    ("ai","")]
)

question_answer_chain = create_stuff_documents_chain(model, template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain) # here 1st chunks are retrived and then it was combined with prompt to get response from llm


user_input = st.chat_input("Type your message:")

if user_input:
    # Append user input to session state as a message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Calculate the length of the user input
    try:
    # Generate system's response (length of the input)
        system_response1 = rag_chain.invoke({"input":user_input})
        system_response=system_response1["answer"]
    except:
        system_response="Some Error at Groq Internal Server" # some times groq.InternalServerError 503 error may occur

    # Append the system's response to session state
    st.session_state.messages.append({"role": "system", "content": system_response})

    # Display updated chat history
    display_messages()


if len(st.session_state.messages) > 0:
    if st.sidebar.button("Clear History"):
        st.session_state.messages = []  # Reset the chat history
else:
    st.sidebar.write("No history to clear")

if len(st.session_state.messages) > 0:
    # Generate the PDF
    pdf_file = generate_pdf()

    # Show the download button only if there is chat history
    st.sidebar.download_button(
        label="Download History",
        data=pdf_file,
        file_name="chat_history.pdf",
        mime="application/pdf",
        key="download_pdf_button"
    )
else:
    # Display a message when there is no chat history
    st.sidebar.write("No history to download")


