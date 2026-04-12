from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from lexicalrichness import LexicalRichness
import textstat
import nltk
from nltk.tokenize import sent_tokenize

# download tokenizer (jalan sekali di server)
nltk.download('punkt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

        # 🔹 NLTK sentence tokenize
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)

        lex = LexicalRichness(text)
        word_count = int(lex.words)

        avg_sentence_length = round(
            word_count / sentence_count, 2
        ) if sentence_count > 0 else 0

        response = {
            "words": word_count,
            "sentences": sentence_count,
            "avg_sentence_length": avg_sentence_length,

            "types": int(lex.terms),
            "ttr": round(float(lex.ttr), 2),
            "rttr": round(float(lex.rttr), 2),
            "cttr": round(float(lex.cttr), 2),
            "mtld": round(float(lex.mtld()), 2) if lex.mtld() else 0,
            "msttr": round(float(lex.msttr(segment_window=msttr_window)), 2),
            "mattr": round(float(lex.mattr(window_size=mattr_window)), 2),
            "hdd": round(float(lex.hdd(draws=hdd_draws)), 2),

            "flesch_reading_ease": round(float(textstat.flesch_reading_ease(text)), 2),
            "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 2),
            "gunning_fog": round(float(textstat.gunning_fog(text)), 2),
            "smog_index": round(float(textstat.smog_index(text)), 2),
        }

        return response

    except Exception as e:
        return {"error": str(e)}