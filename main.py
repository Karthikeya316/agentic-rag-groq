#Document question and answering using RAG
#---------------------------------------------------------------------------
#imports
#---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from markitdown import MarkItDown
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
    md_converter = MarkItDown()  # initialize MarkItDown once
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
            try:
                result = md_converter.convert(file_path)  # MarkItDown converts PDF to markdown
                text = result.text_content
            except Exception as e:
                print(f"MarkItDown failed for {file_name}: {e}, falling back to pypdf")
                reader=PdfReader(file_path)   
                for page in reader.pages:     #loop through all the pages of PDF
                    page_text=page.extract_text()  
                    if page_text:
                        text+=page_text    #extract text from each
        if text.strip():   #check if there is any text
            documents.append({"filename":file_name,"text":text})
   
#---------------------------------------------------------------------------------
#split documents into chunks
#---------------------------------------------------------------------------------
def split_documents(chunk_size=500,overlap=100):
    for document in documents:
        text=document['text']
        source=document['filename']
        step=chunk_size-overlap  #number of characters to skip between  
#Why overlap matters: If an answer spans the boundary between two chunks,
#the overlap ensures it's fully captured in at least one chunk — without it you'd lose context at chunk boundaries.        
        for i in range(0,len(text),step):
            chunk=text[i:i+chunk_size]
            if chunk.strip():    #check if there is any text
                chunks.append({"content":chunk,"source":source})                                
 #-----------------------------------------------------------------------------------
 # TF-IDF Vectors
 # ----------------------------------------------------------------------------------
def create_vectors():
    global vectorizer  #applyingglobal vectorizer
    global chunk_vectors  #applying chunk_vectors
    texts=[chunk['content'] for chunk in chunks]  #extracting container
    vectorizer=TfidfVectorizer(sublinear_tf=True)
    chunk_vectors=vectorizer.fit_transform(texts)
#-------------------------------------------------------------------------------------
# Retreive relevant chunks
#-------------------------------------------------------------------------------------
def retrieve_chunks(query, top_k=3):
    query_vector = vectorizer.transform([query])  # convert query string into TF-IDF vector
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()  # calculate similarity score between query and every chunk vector, flatten to 1D array
    ranked_indices = similarities.argsort()[::-1]  # sort indices by similarity score in descending order (highest first)
    threshold = 0.10  # minimum similarity score to consider a chunk relevant
    best_per_doc = {}  # store best matching chunk per document to avoid returning multiple chunks from same source
    for idc in ranked_indices:
        score = similarities[idc]  # get similarity score for current chunk
        if score < threshold:  # stop if remaining chunks are below relevance threshold
            break
        chunk = chunks[idc]  # get the actual chunk dictionary at this index
        source = chunk["source"]  # get the source filename of this chunk
        if source not in best_per_doc:  # only keep the best chunk per document (first one is best since we sorted descending)
            best_per_doc[source] = {"content": chunk["content"], "score": score, "source": source}  # store chunk content and score
    selected = sorted(best_per_doc.values(), key=lambda x: x['score'], reverse=True)[:top_k]  # sort by score and return top_k results
    return selected  # returns list of top_k most relevant chunks across documents

#---------------------------------------------------------------------------------------
#Agent Decision function
#---------------------------------------------------------------------------------------
def decide_action(query):
    calculation_keywords=["calculate","compute","+","*","/"]
    query_lower=query.lower()
    for keyword in calculation_keywords:
        if keyword in query_lower:
            return "calculation"
    return "document_search" 

#calculator tool
def calculate(query):
    expression=query.lower()
    expression=expression.replace("calculate","")
    try:
        result=eval(expression)
        return f"calculate result:{result}"
    except:
        return "error"
    
#pronoun detection
pronouns={
    "he","she","it","they","his","her","their"
}       

#-----------------------------------------------------------------------------------------
#build search query
#-----------------------------------------------------------------------------------------
def resolve_query(query):
    words = query.lower().split()
    if chat_history and any(word in pronouns for word in words):

        last_answer= chat_history[-1]
        last_answer_text=last_answer.replace("Assistant: ","",1)
        first_sentence=(last_answer_text.split(".")[0]).strip()
        return first_sentence+" "+query
    return query
#-----------------------------------------------------------------------------------------
#ask groq llm
#------------------------------------------------------------------------------------------
def answer_question(query):
    resolved_query = resolve_query(query)
    relevant_chunks = retrieve_chunks(resolved_query)

    if not relevant_chunks:
        return "I couldn't find relevant information in the documents.", "N/A"

    context = "\n\n".join([f"Source: {c['source']}\n{c['content']}" for c in relevant_chunks])

    prompt = f"""You are a helpful assistant answering questions from a document.

If the context contains an MCQ question matching the user's query:
- Return the correct answer option with its text

If the context does not contain a matching MCQ but has relevant information:
- Answer descriptively based on the context

If the context has no relevant information at all:
- Say "I couldn't find relevant information in the documents."
IMPORTANT: Do NOT include SOURCE: inside the ANSWER field. Only use the exact format below.

Respond in this exact format:
ANSWER: <your answer>
SOURCE: <source filename>

Context:
{context}

Question: {resolved_query}"""
    completion=client.chat.completions.create(
        model="llama-3.1-8b-instant" ,
        messages=[{"role":"user",
                "content":prompt}],temperature=0
    )
    raw=completion.choices[0].message.content.strip()
    answer = "No answer found."
    source = "Unknown"
    for line in raw.splitlines():
        if line.startswith("ANSWER:"):
            answer=line[len("ANSWER:"):].strip()
        elif line.startswith("SOURCE:"):
            source=line[len("SOURCE:"):].strip()
    return answer,source


#main program
print("\nLoading Documents...........\n")
load_documents()
if len(documents)==0:
    print("No documents found.")
    exit()
split_documents()
create_vectors()
print("assistant is ready to answer questions")    


#chat loop
while True:
    query=input("Ask Question: ").strip()
    if not query:  # add this
        continue
    if query.lower() in ["quit","exit"]:
        break
    action=decide_action(query)
    if action=="calculation":
        result=calculate(query)
        print(result)
        continue

    answer, source = answer_question(query)  
    chat_history.append(f"Assistant: {answer}")
    print(f"\nAnswer: {answer}")
    print(f"Source: {source}\n")

