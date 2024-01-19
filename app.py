import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch  
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from constants import CHROMA_SETTINGS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


device = torch.device('cpu')

def data_ingestion(query):
    chromedriver_path = 'chromedriver.exe'
    service = Service(chromedriver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-extensions')
    # Adding argument to disable the AutomationControlled flag 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    
    # Exclude the collection of enable-automation switches 
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    
    # Turn-off userAutomationExtension 
    options.add_experimental_option("useAutomationExtension", False) 
    
    # Setting the driver path and requesting a page 
    driver = webdriver.Chrome(options=options) 
    driver = webdriver.Chrome(service=service, options=options)
    query = query.split(" ")
    searchable_query = '' 
    for word in query:
        searchable_query += "+"+word
    url = 'https://www.google.com/search?q='+searchable_query
    archieve_url = 'https://archive.is/'
    driver.get(url)

    news_results = driver.find_elements(By.CSS_SELECTOR, 'div#rso > div >div>div>div')
    news_links = []
    for news_div in news_results:
        try:
            if len(news_links) == 5:
                break
            news_link = news_div.find_element(By.TAG_NAME, 'a').get_attribute('href')
            news_links.append(news_link)
        except Exception as e:
            print("No Elems")
    print(news_links)

    archieved_urls = []
    for link in news_links:
        try:
            driver.get(archieve_url)
            driver.find_element(By.ID, 'url').send_keys(link)
            driver.find_element(By.XPATH,'//*[@id="submiturl"]/div[3]/input').submit()
            archieved_urls.append(driver.current_url)
        except Exception as e:
            print("No Elems")

    news_content = ''
    for url in archieved_urls:
        try:
            driver.get(url)
            element = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.ID, "CONTENT")))
            content = driver.find_element(By.ID,'CONTENT')
            news_content += ' '+ content.text
        except Exception as e :
            print("No Elms")
    driver.quit()

    return news_content

def create_vector_db(raw_content):
    persist_directory = "db"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_text(raw_content)
    texts = text_splitter.create_documents(texts)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

@st.cache_resource
def llm_pipeline():
    checkpoint = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map=device,
        torch_dtype=torch.float32,
        max_length = 1024,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
    )
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# @st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa
    
def process_answer(instruction):
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


def main(): 
    st.set_page_config(
        page_title="Financial News Analyst", 
        page_icon="💸")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Financial News Analyst 💸")

    question = st.text_input("Ask question about current financial news: ")
        
    if question:
        # get news content
        raw_content = data_ingestion(question)

        # create the text chunks
        ingested_data = create_vector_db(raw_content)

        # create conversation chain
        st.session_state.conversation = process_answer(question)
        
    st.session_state.conversation

if __name__ == '__main__':
    main()