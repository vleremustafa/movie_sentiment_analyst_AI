from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
import numpy as np
from lime.lime_text import LimeTextExplainer
from datetime import datetime
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- ADD THIS
# ... (keep your other imports)

app = FastAPI()

# --- NEW: ADD THIS SECTION TO ALLOW COMMUNICATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- 1. AI MODEL & SECURITY SETUP ---
# Loading the sentiment model
clf = pipeline("sentiment-analysis", model="vleramm/sentiment-model", top_k=None)
explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

# FIX: This configuration bypasses the Python 3.12/Passlib 72-byte truncation bug
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__ident="2b" 
)

# --- 2. DATA MODELS ---
class ReviewRequest(BaseModel):
    movie: str
    review: str
    username: str

class UserAuth(BaseModel):
    username: str
    password: str

def predict_probs(texts):
    outputs = clf(texts)
    probs = []
    for out in outputs:
        # Sort labels to ensure [Negative, Positive] order for LIME
        sorted_out = sorted(out, key=lambda x: x['label'])
        probs.append([sorted_out[0]['score'], sorted_out[1]['score']])
    return np.array(probs)

# --- 3. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    # Table for user credentials
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, hashed_password TEXT)''')
    # Table for reviews (linked by 'owner' column)
    c.execute('''CREATE TABLE IF NOT EXISTS reviews 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  owner TEXT, date TEXT, movie TEXT, sentiment TEXT, confidence TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 4. AUTHENTICATION ENDPOINTS ---

@app.post("/signup")
async def signup(user: UserAuth):
    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    try:
        # Securely hash the password using the fixed context
        hashed = pwd_context.hash(user.password)
        c.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (user.username, hashed))
        conn.commit()
        return {"message": "User created"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")
    except Exception as e:
        print(f"Signup Error Traceback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserAuth):
    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    db_user = c.execute("SELECT * FROM users WHERE username = ?", (user.username,)).fetchone()
    conn.close()
    
    if db_user and pwd_context.verify(user.password, db_user[1]):
        return {"status": "success", "username": user.username}
    raise HTTPException(status_code=401, detail="Invalid username or password")

# --- 5. MOVIE DIARY ENDPOINTS (CRUD) ---

@app.post("/analyze")
async def analyze_review(data: ReviewRequest):
    # 1. Get sentiment
    raw_results = clf(data.review)[0]
    best_res = max(raw_results, key=lambda x: x['score'])
    label = best_res['label']
    conf = f"{best_res['score']:.2%}"
    
    # 2. Save to Database
    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    try:
        c.execute("""INSERT INTO reviews (owner, date, movie, sentiment, confidence) 
                     VALUES (?, ?, ?, ?, ?)""", 
                  (data.username, datetime.now().strftime("%Y-%m-%d %H:%M"), data.movie, label, conf))
        conn.commit()
    finally:
        conn.close()

    # 3. Generate LIME Explanation (The Fix)
    # Get all available labels from the explainer's local_exp keys to avoid KeyError
    exp = explainer.explain_instance(data.review, predict_probs, num_features=10, num_samples=100)
    
    # Automatically pick the first available label index to explain
    available_labels = list(exp.local_exp.keys())
    label_to_explain = available_labels[0] if available_labels else 0
    
    return {
        "sentiment": label,
        "confidence": conf,
        "explanation_html": exp.as_html(labels=[label_to_explain])
    }

@app.get("/history/{username}")
async def get_history(username: str):
    conn = sqlite3.connect('movie_diary.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("SELECT * FROM reviews WHERE owner = ? ORDER BY id DESC", (username,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.delete("/delete/{review_id}")
async def delete_review(review_id: int):
    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    c.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()
    return {"message": "Deleted"}

@app.put("/update/{review_id}")
async def update_review(review_id: int, new_movie_name: str, new_review_text: str):
    # Re-analyze sentiment for updated text
    raw_results = clf(new_review_text)[0]
    best_res = max(raw_results, key=lambda x: x['score'])
    new_label = best_res['label']
    new_conf = f"{best_res['score']:.2%}"

    conn = sqlite3.connect('movie_diary.db')
    c = conn.cursor()
    c.execute("""UPDATE reviews SET movie = ?, sentiment = ?, confidence = ? WHERE id = ?""", 
              (new_movie_name, new_label, new_conf, review_id))
    conn.commit()
    conn.close()
    return {"message": "Update successful"}