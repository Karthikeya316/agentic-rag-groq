#Document question and answering using RAG
#---------------------------------------------------------------------------
#imports
#---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
#-----------------------------------------------------------------------------
#load API Key
#-----------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY") #fetch API Key from .env file
if not GROQ_API_KEY:
    print("GROQ_API_KEY NOT FOUND")
    exit() #Stop the program,if the key doesn't exist
#------------------------------------------------------------------------------
#creates groq client
client=Groq(api_key=GROQ_API_KEY)
#-------------------------------------------------------------------------------
#global variables
#-------------------------------------------------------------------------------
document_folder="documents"#folder contains pdf and txt files
documents=[] #store complete documents
chunks=[] #store chunks of document
chat_history=[] #store chat history
vectorizer=None #Vectorize the doc
chunk_vectors=None
#---------------------------------------------------------------------------------
#load documents
#---------------------------------------------------------------------------------
def load_documents():  #os.listdir() returns a list of all filenames in the folder
    for file_name in os.listdir(document_folder):   #loop through all the files
        #Joins the folder path and filename into a full path — e.g. docs/report.pdf
        file_path=os.path.join(document_folder,file_name)  #create a full path
        text=""  
        if file_name.endswith(".txt"):   #read .txt file
            with open(file_path,"r",encoding='utf-8') as file:
                text=file.read()
            #create a PdfReader object, loop through every page, extract text from each page, and append it to text.
            #The if page_text check skips blank/image-only pages that return None         
        elif file_name.endswith(".pdf"):  #read .pdf files
            reader=PdfReader(file_path)   
            for page in reader.pages:     #loop through all the pages of PDF
                page_text=page.extract_text()  
                if page_text:
                    text+=page_text    #extract text from each
        if text.strip():   #check if there is any text
            documents.append({"filename":file_name,"text":text})
                              
            
