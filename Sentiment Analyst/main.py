from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
from lime.lime_text import LimeTextExplainer
from datetime import datetime
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware


# Importojmë menaxherin e databazës (OOP)
from database import DatabaseManager 

app = FastAPI()
db = DatabaseManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. AI SETUP ---
clf = pipeline("sentiment-analysis", model="vleramm/sentiment-model", top_k=None)
explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2b")

# --- 2. DATA MODELS ---
class ReviewRequest(BaseModel):
    movie: str
    review: str
    username: str

class UserAuth(BaseModel):
    username: str
    password: str

# --- 3. HELPER FUNCTIONS ---
def predict_probs(texts):
    outputs = clf(texts)
    probs = []
    for out in outputs:
        sorted_out = sorted(out, key=lambda x: x['label'])
        probs.append([sorted_out[0]['score'], sorted_out[1]['score']])
    return np.array(probs)

# --- 4. WEB SCRAPING ENDPOINT (Zëvendëson Google me DuckDuckGo) ---

# --- 5. AUTHENTICATION ENDPOINTS ---
@app.post("/signup")
async def signup(user: UserAuth):
    try:
        hashed = pwd_context.hash(user.password)
        db.create_user(user.username, hashed)
        return {"message": "User created"}
    except Exception:
        raise HTTPException(status_code=400, detail="Username exists or error occurred.")

@app.post("/login")
async def login(user: UserAuth):
    db_user = db.get_user(user.username)
    if db_user and pwd_context.verify(user.password, db_user['hashed_password']):
        return {"status": "success", "username": user.username}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# --- 6. CORE LOGIC (ANALYZE, HISTORY, CRUD) ---
@app.post("/analyze")
async def analyze_review(data: ReviewRequest):
    raw_results = clf(data.review)[0]
    best_res = max(raw_results, key=lambda x: x['score'])
    label, conf = best_res['label'], f"{best_res['score']:.2%}"
    
    db.save_review(data.username, datetime.now().strftime("%Y-%m-%d %H:%M"), data.movie, label, conf)

    exp = explainer.explain_instance(data.review, predict_probs, num_features=10, num_samples=100)
    available_labels = list(exp.local_exp.keys())
    label_to_explain = available_labels[0] if available_labels else 0
    
    return {
        "sentiment": label, "confidence": conf,
        "explanation_html": exp.as_html(labels=[label_to_explain])
    }

@app.get("/history/{username}")
async def get_history(username: str):
    return db.get_history(username)

@app.delete("/delete/{review_id}")
async def delete_review(review_id: int):
    db.delete_review(review_id)
    return {"message": "Deleted"}

@app.put("/update/{review_id}")
async def update_review(review_id: int, new_movie_name: str, new_review_text: str):
    res = clf(new_review_text)[0]
    best = max(res, key=lambda x: x['score'])
    db.update_review(review_id, new_movie_name, best['label'], f"{best['score']:.2%}")
    return {"message": "Update successful"}