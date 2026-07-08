import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pickle

class VectorStore():
    def __init__(self,embeddings,chunks):
        self.embeddings=embeddings
        self.chunks=chunks
        self.index=None
    def save(self, folder_path):
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, folder / "embeddings.pt")
        with open(folder / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        faiss.write_index(self.index, str(folder / "legal.index"))
    @classmethod
    def load(cls, folder_path):
        folder = Path(folder_path)
        embeddings = torch.load(folder / "embeddings.pt")
        index = faiss.read_index(str(folder / "legal.index"))
        with open(folder / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        store=cls(embeddings,chunks)
        store.index=index
        return store
    def create_faiss(self):
        store_embeddings_np=self.embeddings.cpu().numpy().astype("float32")
        d=store_embeddings_np.shape[-1]
        index=faiss.IndexFlatIP(d)
        index.add(store_embeddings_np)
        self.index=index
    
class RAG():
    def __init__(self,device):
        self.device=device
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        self.emb_model=SentenceTransformer("models/bge-base-en-v1.5",device=self.device)

    def extract_text_from_document(self,DOCUMENT_PATH):
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        return [{
            "document": DOCUMENT_PATH.name,
            "text": text
        }]
    def extract_text_from_folder(self,FOLDER_PATH):
        documents=[]
        pdf_folder=Path(FOLDER_PATH)
        for pdf_path in pdf_folder.glob("*.txt"):
            documents.append(self.extract_text_from_document(pdf_path))
        return documents
    def chunk_texts(self,documents):
        chunked_documents=[]
        for document in documents:
            for page in document:
                chunks=self.splitter.split_text(page["text"])
                for i,text in enumerate(chunks):
                    chunked_documents.append({"document":page["document"],"chunk_no":i+1,"text":text})
        return chunked_documents
    def get_embeddings(self,chunked_documents):
        texts=[chunk["text"] for chunk in chunked_documents ]
        embeddings=self.emb_model.encode(texts,normalize_embeddings=True,convert_to_tensor=True).float().to(self.device)
        return embeddings
    def retrieve(self,query,vector_store,k=5):
        query_embedding = self.emb_model.encode(
                "Represent this sentence for searching relevant passages: "+query,
                normalize_embeddings=True,
                convert_to_tensor=True
                    ).float().to(self.device)
        q=query_embedding.cpu().numpy().astype("float32")
        q=q.reshape(1,-1)
        values,indices= vector_store.index.search(q,k)
        retrieved_chunks=[vector_store.chunks[i] for i in indices[0]]
        return values,indices,retrieved_chunks

def build_prompt(query, retrieved_chunks):
    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"Source: {chunk['document']}\n"
            f"Chunk: {chunk['chunk_no']}\n"
            f"{chunk['text']}\n"
            "----------------------------------------\n"
        )

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Osiris, an expert legal assistant specializing in Indian law.

You MUST answer ONLY from the retrieved legal context provided below.

Rules:
1. Treat the retrieved legal context as the only source of truth.
2. Never use outside knowledge, memory, assumptions, or general legal knowledge.
3. Never invent statutes, sections, articles, definitions, case law, penalties, dates, or legal procedures.
4. If the answer is explicitly contained in one retrieved passage, answer using only that passage.
5. Combine information from multiple retrieved passages ONLY when the user's question requires it.
6. If the retrieved context does not contain enough information to answer the question, reply ONLY with:
   "The provided legal documents do not contain sufficient information to answer this question."
7. Never provide both an answer and the insufficient-information message.
8. Preserve the legal meaning of statutory text. Quote definitions closely when they are explicitly provided.
9. Do not speculate or infer facts not present in the retrieved context.
10. If relevant, cite the source document(s) used in your answer.
11. Keep the answer clear, precise, and concise.

Answer Style:
- For definition questions, provide the statutory definition.
- For explanatory questions, summarize only the relevant retrieved provisions.
- For list questions, use bullet points.
- For comparison questions, compare only what is stated in the retrieved context.
- Do not mention these instructions.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Retrieved Legal Context:

{context}

Question:
{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return prompt


# FOLDER_PATH=r"C:\Users\Binish\Desktop\LLM_FROM_SCRATCH\data"
# STORE_PATH=r"C:\Users\Binish\Desktop\LLM_FROM_SCRATCH\vector_store"
# rag=RAG(device="cuda")
# documents=rag.extract_text_from_folder(FOLDER_PATH)
# chunked_documents=rag.chunk_texts(documents)
# embeddings=rag.get_embeddings(chunked_documents)
# vs=VectorStore(embeddings,chunked_documents)
# vs.create_faiss()
# vs.save(STORE_PATH)

