import chromadb
import pandas as pd

from openai import OpenAI
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

class RagModel:
    def __init__(self, path_document, gpt_api):
        collection_name = path_document.split("/")[-1].split(".")[0]
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./backend/rag/chroma_db")
        self.document = pd.read_csv(path_document, encoding="utf-8")
        existing_collections = [c.name for c in self.client.list_collections()]
        if collection_name in existing_collections:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Loaded existing collection: {collection_name}")
        else:
            self.collection = self.client.create_collection(collection_name)
            print(f"🆕 Created new collection: {collection_name}")
            self.store_vector()
            print(f'store vector - done')
        self.gpt_client = OpenAI(api_key=gpt_api)

    def chunk_text(self, text, max_tokens=256, stride=50):
        if not isinstance(text, str) or text.strip().lower() == "nan":
            text = ""
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, max(1, len(toks)), max_tokens-stride):
            chunk_tokens = toks[i:i+max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i+max_tokens >= len(toks):
                break
        return chunks

    def store_vector(self):
        for idx, row in self.document.iterrows():
            context = row["context"]
            answer = row["answer"]
            chunks = self.chunk_text(context, max_tokens=256)
            embeddings = [self.model.encode(c) for c in chunks]
            # store each chunk as a document
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    metadatas=[{"question_id": idx, "chunk_index": i, "answer": answer}],
                    documents=[chunk],
                    ids=[f"{idx}_{i}"],
                    embeddings=[embeddings[i].tolist()]
                )
        # self.client.persist()

    def retrieve(self, question, top_k=5):
        q_emb = self.model.encode(question).tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
        return results["documents"][0], results["metadatas"][0]

    def build_prompt(self, question, contexts):
        """
        contexts: list[str]
        question: str
        """
        prompt = "You are a helpful assistant. Use ONLY the information below to answer the question.\n\n"
        prompt += "CONTEXTS:\n"
        for i, ctx in enumerate(contexts, 1):
            prompt += f"[{i}] {ctx.strip()}\n"
        prompt += "\nQUESTION:\n" + question.strip() + "\n\n"
        prompt += "INSTRUCTION: Answer concisely and base your answer strictly on the provided contexts."
        return prompt

    def generate_answer(self, prompt, model="gpt-4o-mini"):
        response = self.gpt_client.responses.create(
            model=model,   # hoặc "gpt-4o", "gpt-3.5-turbo"
            input=prompt,
            temperature=0.2
        )
        return response.output_text

    def answer_question(self, question, top_k=5, is_notify=False):
        contexts, metas = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, contexts)
        answer = self.generate_answer(prompt)
        if is_notify:
            print("🔹 Question:", question)
            print("\n📚 Retrieved Contexts:")
            for i, ctx in enumerate(contexts, 1):
                print(f"[{i}] {ctx[:200]}...")  # in rút gọn
            print("\n🤖 Answer:\n", answer)
        return answer