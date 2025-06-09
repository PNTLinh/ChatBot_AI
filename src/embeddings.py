from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import os, shutil

STORE_DIR = "/content/drive/MyDrive/Colab Notebooks/ChatBot_AI/data/faiss_store"
def build_or_load_embeddings(docs, metas, store_dir=STORE_DIR):
    idx = os.path.join(store_dir, "index.faiss")
    ds  = os.path.join(store_dir, "docstore.pkl")
    if os.path.isdir(store_dir) and (not os.path.isfile(idx) or not os.path.isfile(ds)):
        shutil.rmtree(store_dir)
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.isdir(store_dir):
<<<<<<< HEAD
        print("Building FAISS index…")
        vs = FAISS.from_texts(texts=docs, embedding=embed, metadatas=metas)
        vs.save_local(store_dir)
        print("Saved to Drive.")
        return vs

    print("Loading FAISS index from Drive…")
=======
        print("▶ Building FAISS index…")
        vs = FAISS.from_texts(texts=docs, embedding=embed, metadatas=metas)
        vs.save_local(store_dir)
        print("✔️ Done, saved to Drive.")
        return vs

    print("▶ Loading FAISS index from Drive…")
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
    return FAISS.load_local(store_dir, embed)