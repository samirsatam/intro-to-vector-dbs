import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("Hello from intro-to-vector-dbs!")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)

    # Madison wrote 49–58
    # John Jay wrote 64
    query = "Who wrote Federalist No. 64?"
    # Madison and Hamilton wrote 18-20
    query = "Who wrote Federalist No. 18?"
    # query = "Give details about the paper titled 'Harry Potter'"
    # query = "What paper mentions the text - 'The internal effects of a mutable policy are still more calamitous'"

    vectorstore = PineconeVectorStore(
        index_name=os.environ["FEDERALIST_PAPERS_INDEX_NAME"], embedding=embeddings
    )

    # custom prompt
    template = """Use the following pieces of context to answer the question at the end.

    {context}

    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    You are a linguistic expert, and a historian. You can analyze patterns in text and attribute them to a specific person or writer.
    You also know that the writers of the Federalist papers are John Jay, Alexander Hamilton, and James Madison.
    You are an expert historian as well, and know a lot about the writing styles of John Jay, Alexander Hamilton, and James Madison.
    Sometimes a Federalist paper has a single author or it is collaboration between multiple authors.
    You will follow the steps below to answer the question at the end.
    1. Figure out if the Authorship is ambiguous or not.
    2. If there is no ambiguity about the authorship, just output the author or authors as is.
    3. Make sure to list primary author, secondary author and and tertiary author if any.
    4. If there is ambiguity about the authorship, use your expertise about the style of writing and analyze the text to resolve the ambiguity and determine the correct author.
    5. Describe the steps you followed to answer the question.
    
    Question: {question}
    Helpful Answer:"""

    #make the text into a prompt
    custom_rag_prompt = PromptTemplate.from_template(template=template)

    combine_docs_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    res = retrieval_chain.invoke(input={"input": query, "question": query})

    # make up the RAG chain, format_docs == all the documents appended together.
    # rag_chain = (
    #         {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    #         | custom_rag_prompt
    #         | llm
    # )
    #
    # res = rag_chain.invoke(query)
    print(f'Context: {res["context"]}\n\n')
    print(res['answer'])


if __name__ == "__main__":
    main()
