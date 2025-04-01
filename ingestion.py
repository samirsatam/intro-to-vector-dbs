import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    pathlist = Path("./federalist_papers").iterdir()
    for path in pathlist:
        if path.is_file():
            loader = TextLoader(path)
            document = loader.load() # loads in a LangChain document.

            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            embeddings = OpenAIEmbeddings()
            text_splitter = SemanticChunker(embeddings=embeddings)
            texts = text_splitter.split_documents(document)
            print(f"created {len(texts)} chunks")

            PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["FEDERALIST_PAPERS_INDEX_NAME"])
            print("Finished")
