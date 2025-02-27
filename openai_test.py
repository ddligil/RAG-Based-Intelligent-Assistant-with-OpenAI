from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
import logging
import hashlib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


logger.info("Creating splitter...")
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
logger.info("Splitter created.")


logger.info("Creating embeddings...")
embedding_model = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="bge-m3"
)
logger.info("Embeddings created.")



logger.info("Checking for existing vector store...")
vector_store = Chroma(persist_directory='db', embedding_function=embedding_model)
logger.info("Vector store loaded or created.")



def get_document_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

logger.info("Processing documents...")
pdf_path = './example.pdf'
documents = PyPDFLoader(file_path=pdf_path).load_and_split(text_splitter=splitter)
logger.info(f"Processed {len(documents)} documents.")


existing_doc_hashes = set(get_document_hash(doc) for doc in vector_store.get()['documents'])
new_documents = [doc for doc in documents if get_document_hash(doc.page_content) not in existing_doc_hashes]

if new_documents:
    logger.info(f"Adding {len(new_documents)} new documents to the vector store...")
    vector_store.add_documents(new_documents)
    logger.info("New documents added.")
else:
    logger.info("No new documents to add.")