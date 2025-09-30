import streamlit as st
import joblib
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ------------------ Preprocess ------------------
def preprocess_text(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)  # tách từ bằng regex
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ------------------ Giao diện chính ------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

st.title("📊 Phân loại cảm xúc bình luận")
st.write("Ứng dụng demo sử dụng **NLP + Machine Learning** để phân loại review tích cực / tiêu cực.")

# Sidebar cho các lựa chọn
st.sidebar.header("⚙️ Cấu hình")
method = st.sidebar.selectbox("Chọn phương pháp:", ["Bag of Words (BoW)", "TF-IDF"])
model = st.sidebar.selectbox("Chọn mô hình:", ["Naive Bayes", "Logistic Regression"])
show_preprocess = st.sidebar.checkbox("Hiển thị văn bản sau khi xử lý")

user_input = st.text_area("✍️ Nhập review:", "", height=150)

models = {
    "NaiveBayes": {
        "BoW": "naive_bayes_bow.pkl",
        "TF-IDF": "naive_bayes_tfidf.pkl"
    },
    "LogisticRegression": {
        "BoW": "logistic_regression_bow.pkl",
        "TF-IDF": "logistic_regression_tfidf.pkl"
    }
}
vectorizers = {
    "BoWVectorizer": "bow_vectorizer.pkl",
    "TF-IDFVectorizer": "tfidf_vectorizer.pkl"
}

if st.button("Dự đoán"):
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập review trước khi dự đoán!")
    else:
        with st.spinner("🔎 Đang phân tích..."):
            clean_text = preprocess_text(user_input)

            if model == "Naive Bayes":
                if method == "Bag of Words (BoW)":
                    vectorizer = joblib.load(vectorizers["BoWVectorizer"])
                    model = joblib.load(models["NaiveBayes"]["BoW"])
                    vectorized_text = vectorizer.transform([clean_text])
                else:  # TF-IDF
                    vectorizer = joblib.load(vectorizers["TF-IDFVectorizer"])
                    model = joblib.load(models["NaiveBayes"]["TF-IDF"])
                    vectorized_text = vectorizer.transform([clean_text])
                prediction = model.predict(vectorized_text)[0]
            else:  # Logistic Regression
                if method == "Bag of Words (BoW)":
                    vectorizer = joblib.load(vectorizers["BoWVectorizer"])
                    model = joblib.load(models["LogisticRegression"]["BoW"])
                    vectorized_text = vectorizer.transform([clean_text])
                else:  # TF-IDF
                    vectorizer = joblib.load(vectorizers["TF-IDFVectorizer"])
                    model = joblib.load(models["LogisticRegression"]["TF-IDF"])
                    vectorized_text = vectorizer.transform([clean_text])
                prediction = model.predict(vectorized_text)[0]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("😡 Review **tiêu cực!**")
            else:
                st.success("😊 Review **tích cực!**")
        with col2:
            st.metric("Độ dài review (sau xử lý)", f"{len(clean_text.split())} từ")

        if show_preprocess:
            st.info(f"🔧 Văn bản sau khi xử lý: \n\n`{clean_text}`")
