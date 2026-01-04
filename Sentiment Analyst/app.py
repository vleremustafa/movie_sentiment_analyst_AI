import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Movie Diary", layout="wide")

# URL e Backend-it tuaj FastAPI
API_URL = "http://127.0.0.1:8000"

# --- 1. SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None
if "auto_review" not in st.session_state:
    st.session_state.auto_review = ""

# --- 2. SIDEBAR: AUTHENTICATION & SETTINGS ---
with st.sidebar:
    st.title("👤 User Portal")
    
    if not st.session_state.logged_in:
        auth_mode = st.radio("Choose Action:", ["Login", "Sign Up"])
        user_input = st.text_input("Username", key="auth_user")
        pass_input = st.text_input("Password", type="password", key="auth_pass")
        
        if auth_mode == "Login":
            if st.button("Sign In"):
                try:
                    res = requests.post(f"{API_URL}/login", json={"username": user_input, "password": pass_input})
                    if res.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.username = user_input
                        st.success(f"Welcome back, {user_input}!")
                        st.rerun()
                    else:
                        st.error(res.json().get('detail', "Invalid username or password."))
                except Exception as e:
                    st.error("Backend is offline. Check your Uvicorn terminal.")
        
        else:
            if st.button("Create Account"):
                if user_input and pass_input:
                    try:
                        res = requests.post(f"{API_URL}/signup", json={"username": user_input, "password": pass_input})
                        if res.status_code == 200:
                            st.success("✨ Account created! Please switch to Login.")
                        else:
                            st.error(res.json().get('detail', "Signup failed."))
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
                else:
                    st.warning("Please enter both fields.")
    else:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
    st.divider()
    st.title("⚙️ Settings")
    fast_mode = st.checkbox("🚀 Fast Mode (Skip Explanation)", value=False)

# --- 3. MAIN APP LOGIC (ONLY IF LOGGED IN) ---
if not st.session_state.logged_in:
    st.title("🍿 AI Movie Diary")
    st.info("Please Login or Sign Up from the sidebar to access your private diary.")
else:
    st.title(f"🍿 {st.session_state.username}'s Movie Diary")
    
    # --- CREATE SECTION ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("➕ Add a Review")
        movie_name = st.text_input("Movie Title:", placeholder="e.g., Inception")
        
   

        # Vlera merret nga session_state
        user_review = st.text_area(
            "Review Content:", 
            value=st.session_state.auto_review, 
            placeholder="Write your review or use Auto-Fill...",
            height=200
        )
        
        if st.button("Analyze & Save"):
            if movie_name and user_review:
                with st.spinner("Analyzing..."):
                    payload = {
                        "movie": movie_name, 
                        "review": user_review,
                        "username": st.session_state.username
                    }
                    try:
                        res = requests.post(f"{API_URL}/analyze", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            if not fast_mode and 'explanation_html' in data:
                                custom_css = "<style>body{color:white !important;} .lime.explanation{background-color:#0e1117;} text{fill:white !important;}</style>"
                                st.session_state.last_explanation = custom_css + data['explanation_html']
                            
                            st.session_state.auto_review = "" # Resetojmë fushën pas ruajtjes
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please fill in both fields.")

    with col2:
        if st.session_state.last_explanation:
            st.subheader("🔍 AI Reasoning")
            components.html(st.session_state.last_explanation, height=400, scrolling=True)

    st.divider()

    # --- READ/HISTORY SECTION ---
    st.subheader("📜 Your Watch History")
    
    try:
        hist_res = requests.get(f"{API_URL}/history/{st.session_state.username}")
        if hist_res.status_code == 200:
            df = pd.DataFrame(hist_res.json())
            if not df.empty:
                
                # --- ANALIZA VIZUALE ---
                st.write("### 📊 AI Insights")
                c_chart1, c_chart2 = st.columns(2)
                
                with c_chart1:
                    label_col = 'sentiment' if 'sentiment' in df.columns else 'label'
                    sentiment_counts = df[label_col].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    
                    fig_pie = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                                     title='Overall Movie Sentiment',
                                     color='Sentiment',
                                     color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c'})
                    # Rregulluar për Streamlit 2026
                    st.plotly_chart(fig_pie, width='stretch')

                with c_chart2:
                    fig_hist = px.histogram(df, x="confidence", nbins=10, 
                                            title="AI Prediction Confidence",
                                            color_discrete_sequence=['#3498db'])
                    # Rregulluar për Streamlit 2026
                    st.plotly_chart(fig_hist, width='stretch')
                
                st.divider()

                cols_to_show = [c for c in df.columns if c != 'owner']
                st.dataframe(df[cols_to_show], width='stretch', hide_index=True)

                # --- UPDATE & DELETE (CRUD) ---
                st.write("---")
                st.subheader("🛠️ Manage Records")
                sel_id = st.selectbox("Select ID to Edit/Delete:", df['id'].tolist())
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**✏️ Edit Entry**")
                    up_title = st.text_input("Update Title:", key="up_t")
                    up_text = st.text_area("Update Review:", key="up_v")
                    if st.button("Confirm Update"):
                        if up_title and up_text:
                            params = {"new_movie_name": up_title, "new_review_text": up_text}
                            if requests.put(f"{API_URL}/update/{sel_id}", params=params).status_code == 200:
                                st.success("Updated!")
                                st.rerun()

                with c2:
                    st.write("**🗑️ Delete Record**")
                    if st.button("Delete Permanently", type="primary"):
                        if requests.delete(f"{API_URL}/delete/{sel_id}").status_code == 200:
                            st.warning("Deleted!")
                            st.rerun()
            else:
                st.info("No reviews yet.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")