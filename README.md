# movie_sentiment_analyst_AI
AI-powered Movie Diary: A full-stack sentiment analysis application built with FastAPI, Streamlit, and Hugging Face Transformers. Features user authentication, LIME interpretability, and full CRUD history.

🍿 AI Movie Diary & Sentiment Analyzer
An end-to-end Machine Learning application that logs movie reviews and uses a custom-trained Transformer model to analyze sentiment.

🌟 Project Overview
This project features a sentiment analysis engine trained on the IMDB Dataset (50,000 reviews) from Kaggle. The model was trained using the Hugging Face transformers library and deployed via a FastAPI backend with a Streamlit frontend.
It doesn't just give a "Positive" or "Negative" result—it uses LIME (Local Interpretable Model-agnostic Explanations) to highlight exactly which words influenced the AI's decision, providing full transparency into the model's reasoning.

🛠️ Tech Stack
Model Training: DistilBERT / Transformers (Trained on Kaggle GPU)
Dataset: IMDB Movie Reviews (50k samples)
Backend: FastAPI (Python)
Frontend: Streamlit
Database: SQLite (SQL)
Interpretability: LIME
Visualizations: Plotly Express

✨ Key Features
User Authentication: Secure signup and login with hashed passwords.
Custom NLP Model: High-accuracy sentiment classification.
AI Reasoning: Visual word-level explanations for every prediction.
Personal Diary: Full CRUD (Create, Read, Update, Delete) for your movie history.
Data Export: Download your movie logs as a CSV file.
Analytics: Interactive pie charts showing your overall "Movie Mood."

🚀 How to Run
Clone the repository:
Bash
git clone https://github.com/your-username/sentiment-project.git
cd sentiment-project
Install requirements:

Bash
pip install -r requirements.txt

Start the FastAPI Backend:
Bash
uvicorn main:app --reload

Start the Streamlit Frontend:
Bash
streamlit run app.py

