import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Azhal Fake News Detection",
    page_icon="Azhal.png",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.stTextArea textarea {
    background-color: #262730;
    color: white;
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    text-align: center;
}
.fake {
    background-color: #3b0d0d;
}
.real {
    background-color: #0d3b1f;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("logistic_regression_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

st.title("📰 Fake News Detection System")
st.write("Paste a news article below to check whether it is **Fake** or **Real**.")

news_input = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste your news article here..."
)

if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("⚠ Please enter some news text")
    else:
        cleaned_text = clean_text(news_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]

        confidence = max(probabilities) * 100

        if prediction == 0:
            st.markdown(
                f"""
                <div class="result-box fake">
                    <h2>🚨 Fake News</h2>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(confidence))

        else:
            st.markdown(
                f"""
                <div class="result-box real">
                    <h2>✅ Real News</h2>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(confidence))

