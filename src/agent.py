from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv
from pathlib import Path
import os

class DocumentProcessor:
    # Handles loading and splitting of text documents.

    def __init__(self, file_path: str, chunk_size: int = 750, chunk_overlap: int = 150):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def load_documents(self):
        # Loads documents from a text file.
        try:
            loader = TextLoader(self.file_path)
            return loader.load()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.file_path} not found.")

    def split_documents(self, documents):
        # Splits documents into chunks.
        return self.text_splitter.split_documents(documents)

class VectorStore:
    # Manages creation of a FAISS vector store with embeddings.

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )

    def create_vector_store(self, documents):
        # Creates a FAISS vector store from documents.
        if not documents:
            raise ValueError("No documents provided for vector store creation.")
        return FAISS.from_documents(documents, self.embeddings)

class LanguageModel:
    # Sets up a language model pipeline for text generation.

    def __init__(self, model_name: str = "unsloth/mistral-7b-instruct-v0.1-bnb-4bit"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype="auto",
            device_map="auto",
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=0.2
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def get_llm(self):
        # Returns the language model pipeline.
        return self.llm

class QASystem:
    # Manages the RetrievalQA chain for answering questions with a prompt template.

    def __init__(self, llm, vector_store, chain_type: str = "stuff", k: int = 5):
        # Define a prompt template for structured QA
        prompt_template = """You are a helpful assistant answering questions about NASA and its missions, including the Voyager program.
Use the following context to provide a concise and accurate answer. If the answer is not in the context, say you don't know.

Context: {context}

Question: {question}

Answer: """
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=vector_store.as_retriever(search_kwargs={'k': k}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        # Runs a query and returns the answer.
        if not question or not question.strip():
            return "Please provide a valid question."
        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "I am not able to provide you with an answer.").strip()
            return answer
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Initialize components
    try:
        dotenv_path = Path('config/variables.env')
        load_dotenv(dotenv_path=dotenv_path)

        document_path = os.getenv('DOCUMENT_PATH')
        doc_processor = DocumentProcessor(document_path="data/nasa_info.txt")
        vector_store = VectorStore()
        llm = LanguageModel()

        # Process documents
        documents = doc_processor.load_documents()
        texts = doc_processor.split_documents(documents)

        # Create vector store
        db = vector_store.create_vector_store(texts)

        # Set up QA system
        qa_system = QASystem(llm.get_llm(), db)

        # Interactive console loop
        print("Welcome to the NASA QA System! Type 'exit' or 'quit' to stop.")
        while True:
            question = input("Enter your question: ")
            if question.lower() in ['exit', 'quit']:
                print("Exiting the QA system. Goodbye!")
                break
            if not question.strip():
                print("Please enter a valid question.\n")
                continue
            answer = qa_system.query(question)
            print("Answer:")
            print(answer)
            print("\n")
    except Exception as e:
        print(f"Error initializing QA system: {str(e)}")

if __name__ == "__main__":
    main()
