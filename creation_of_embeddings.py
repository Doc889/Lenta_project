import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def creating_vector_store():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Текущая директория
    files_dir = os.path.join(current_dir, 'files/pdf')  # Директория со всеми файлами
    db_dir = os.path.join(current_dir, 'db')  # Директория с базами данных
    persistent_directory = os.path.join(db_dir, "chroma_db_lenta")  # Директория с векторной базой данных

    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        if not os.path.exists(files_dir):
            raise FileNotFoundError(
                f"The file {files_dir} does not exist. Please chack the path"
            )

        files = [f for f in os.listdir(files_dir) if f.endswith(".pdf")]  # Генератор создания массива со всеми файлами

        documents = []  # Массив для объектов Document
        for file in files:  # Цикл по всем файлам
            file_path = os.path.join(files_dir, file)  # Путь к каждому файлу
            loader = PyMuPDFLoader(file_path)  # Превращение содержимого файлов в объект Document
            files_docs = loader.load()  # Загрузка содержимого в переменную files_docs
            for doc in files_docs:
                doc.metadata = {"source": file}
                documents.append(doc)  # Добавляем в массив объект Document

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(
            documents)  # Делим page_content на куски (chunks) согласно правилам сплиттера (chunk_size, chunk_overlap, и т.д.).

        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")

        docs = [doc for doc in docs if doc.page_content.strip()]
        if not docs:
            raise ValueError("All document chunks are empty. Cannot create embeddings.")

        print("\n--- Creating embeddings ---")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")  # Инициализация объекта HugginFace для создания embedding

        print("\n--- Finished creating embeddings ---")

        print("\n--- Creating and persisting vector store ---")
        db = Chroma.from_documents(docs, embeddings,
                                   persist_directory=persistent_directory)  # Создание векторной базы данных
        print("\n--- Finished creating and persisting vector store ---")

    else:
        print("Vector store already exists. No need to initialize")