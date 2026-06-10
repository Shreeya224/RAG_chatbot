from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
import uuid


class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def add_document(self, text: str, metadata: dict = {}) -> str:
        """Split text into chunks and add to vector store."""
        doc_id = str(uuid.uuid4())[:8]
        metadata["doc_id"] = doc_id

        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

        if self.vectorstore is None:
            # First document — create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing vector store
            self.vectorstore.add_documents(documents)

        print(f"Added {len(chunks)} chunks from document {doc_id}")
        return doc_id

    def query(self, question: str, k: int = 3) -> dict:
        """Retrieve relevant chunks and generate an answer."""
        if self.vectorstore is None:
            return {
                "answer": "No documents in knowledge base yet. Please upload documents first.",
                "sources": []
            }

        # Retrieve top-k relevant chunks
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})

        # Extract source text snippets
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer": result["result"],
            "sources": sources
        }

    def save(self, path: str = "vectorstore"):
        """Save vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")

    def load(self, path: str = "vectorstore"):
        """Load vector store from disk."""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from {path}")
