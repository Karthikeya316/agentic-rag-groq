#Document question and answering using RAG
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

#load API Key
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY") #fetch API Key from .env file
if not GROQ_API_KEY:
    print("GROQ_API_KEY NOT FOUND")
    exit() #Stop the program,if the key doesn't exist

#creates groq client
client=Groq(api_key=GROQ_API_KEY)
#global variables
document_folder="documents" #folder contains pdf and txt files
chunks=[] #store chunks of document
chat_history=[] #store chat history
vectorizer=TfidfVectorizer() #Vectorize the doc
chunk_vectors=None
