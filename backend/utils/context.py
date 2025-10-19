
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextBase:
    def __init__(self):
        with open("backend/utils/contexts.txt", "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f if line.strip()]
        print(f"Đã đọc {len(self.documents)} đoạn thông tin.")

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def search_best_context(self, query, top_k=1):
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Lấy top K kết quả có độ tương đồng cao nhất
        top_indices = cosine_sim.argsort()[-top_k:][::-1][0]
        return self.documents[top_indices]