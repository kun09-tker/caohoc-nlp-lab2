import time

from backend.utils.model import ChatbotModel
from backend.utils.context import ContextBase
from backend.utils.translate import translate
from backend.rag.model import RagModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.aspect_sentiment.model import AspectSentimentClassification

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:5500']
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "google/flan-t5-large"
PATH_DOCUMENT = "backend/rag/raw/intermediate_science.csv"
API_CHATGPT = ""
ASC_TOKENIZER = "backend/checkpoints/LLM_CL/tokenizer"
ASC_CHECKPOINT = "backend/checkpoints/LLM_CL/all_debertaV2_proposal_16_16_0.9417.pt"

c = ContextBase()
tr = translate()
bot = ChatbotModel(MODEL_NAME, "backend\checkpoints\DebertaV3")
rag_model = RagModel(PATH_DOCUMENT, API_CHATGPT)
asc = AspectSentimentClassification(ASC_CHECKPOINT, ASC_TOKENIZER)

def is_model_ready():
    global c, tr, bot
    try:
        rag_model.answer_question("")
        bot.generate_ans("")
        asc.predict_sentiment("eng_review", "eng_aspect")
        return True
    except Exception as e:
        print(e)
        return False

while is_model_ready() == False:
    print("Wait for stable")
print("ok")
# Tải mô hình QA
# qa = pipeline("question-answering", model="allenai/sciq")

@app.post("/ask")
async def ask(request: Request):
    if not is_model_ready():
        return {"answer": "🟡 Hệ thống đang chuẩn bị mô hình RAG, vui lòng chờ..."}
    data = await request.json()
    question = data.get("question", "")
    eng_question = tr.translate_vi_en(question)
    context = c.search_best_context(eng_question)
    fin_question = f"Question: {eng_question} \nContext: {context}"
    answer = bot.generate_ans(fin_question).replace("Answer: ", "").capitalize()
    vi_answer = tr.translate_en_vi(answer)
    return {"answer": f"{answer} - {vi_answer} \n\n" \
                      f"Nội dung tham khảo: {tr.translate_en_vi(context).replace('.', '. ')}"}

@app.post("/ask_rag")
async def ask_rag(request: Request):
    if not is_model_ready:
        return {"answer": "🟡 Hệ thống đang chuẩn bị mô hình RAG, vui lòng chờ..."}
    data = await request.json()
    question = data.get("question", "")
    eng_question = tr.translate_vi_en(question)
    answer = rag_model.answer_question(eng_question, is_notify=False)
    vi_answer = tr.translate_en_vi(answer)
    return {"answer": vi_answer}

@app.post("/predict_sentiment")
async def predict_sentiment(request: Request):
    data = await request.json()
    review = data.get("review", "")
    aspect = data.get("aspect", "")
    eng_review = tr.translate_vi_en(review).lower()
    eng_aspect = tr.translate_vi_en(aspect).lower()
    print(eng_aspect)
    print(eng_review)
    if eng_aspect not in eng_review:
        return "neutral"
    sentiment = asc.predict_sentiment(eng_review, eng_aspect)
    return {"sentiment": sentiment}
