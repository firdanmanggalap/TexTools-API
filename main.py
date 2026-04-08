from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from lexicalrichness import LexicalRichness
import textstat
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau masukkan domain web lu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        text = text.replace("’", "").replace("'", "")

        msttr_window = int(data.get("msttr_window", 50))
        mattr_window = int(data.get("mattr_window", 50))
        hdd_draws = int(data.get("hdd_draws", 42))

        if not text:
            return {"error": "Empty text"}

        lex = LexicalRichness(text)
        response = {
            "words": int(lex.words),
            "types": int(lex.terms),
            "ttr": round(float(lex.ttr), 5),
            "rttr": round(float(lex.rttr), 5),
            "cttr": round(float(lex.cttr), 5),
            "mtld": round(float(lex.mtld()), 5),
            "msttr": round(float(lex.msttr(segment_window=msttr_window)), 5),
            "mattr": round(float(lex.mattr(window_size=mattr_window)), 5),
            "hdd": round(float(lex.hdd(draws=hdd_draws)), 5),
            "flesch_reading_ease": round(float(textstat.flesch_reading_ease(text)), 5),
            "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 5),
            "gunning_fog": round(float(textstat.gunning_fog(text)), 5),
            "smog_index": round(float(textstat.smog_index(text)), 5),
        }
        return response

    except Exception as e:
        return {"error": str(e)}
