import time

from backend.utils.model import ChatbotModel
from backend.utils.context import ContextBase
from backend.utils.translate import translate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:5500']
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "google/flan-t5-large"
c = ContextBase()
tr = translate()
bot = ChatbotModel(MODEL_NAME, "backend\checkpoints\DebertaV3")

def is_model_stable():
    global c, tr, bot
    try:
        bot.generate_ans("")
        return True
    except Exception as e:
        return False

@app.get("/health")
def health_check():
    if is_model_stable():
        return {"status": "ok"}
    else:
        return {"status": "loading"}

# Tải mô hình QA
# qa = pipeline("question-answering", model="allenai/sciq")

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    eng_question = tr.translate_vi_en(question)
    context = c.search_best_context(eng_question)
    fin_question = f"Question: {eng_question} \nContext: {context}"
    answer = bot.generate_ans(fin_question).replace("Answer: ", "").capitalize()
    vi_answer = tr.translate_en_vi(answer)
    return {"answer": f"{answer} - {vi_answer} \n\n" \
                      f"Nội dung tham khảo: {tr.translate_en_vi(context).replace('.', '. ')}"}