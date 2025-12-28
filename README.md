# Fake-News-Detector
This project is a Fake News Detection web application built using Machine Learning and Streamlit.
It allows users to enter a news article and instantly predicts whether the news is Real or Fake, along with a confidence score.

## **Features**
1. Detects Real vs Fake news
2. Uses Logistic Regression with TF-IDF Vectorization
3. Displays prediction confidence
4. Simple and interactive Streamlit UI
5. Lightweight and fast

## **Machine Learning Model**
**Algorithm:** Logistic Regression

**Text Vectorization:** TF-IDF (Term Frequency – Inverse Document Frequency)

**Libraries:** Scikit-learn, Pandas, NumPy

Trained model and vectorizer are saved as .pkl files and loaded into the app for real-time predictions.

## **Tech Stack**
```
Python
Scikit-learn
Streamlit
Pandas
NumPy
NLTK
Pickle
```
## **Project Structure**
```
Fake-News-Detection/
│
|--LICENSE
|--README.md
|--app.py
|--logistic_regression_model.pkl
|--requirements.txt
|--tfidf_vectorizer.pkl
└── train_news.7z
```
## **Installation & Setup**
**1️.Clone the Repository**
```
git clone https://github.com/suryamoorthy006/Fake-News-Detector.git
cd fake-news-detection
```
**2️.Install Dependencies**
```
pip install -r requirements.txt
```
**3️.Run the Streamlit App**
```
streamlit run app.py
```
The app will open in your browser at:
```
http://localhost:8501
```
## **How It Works**

1. User enters a news article
2. Text is cleaned and vectorized using TF-IDF
3. Logistic Regression model predicts the label
4. App displays:
   
       1. Prediction: Real / Fake
       2. Confidence Score
   
## **requirements.txt**
```
streamlit
scikit-learn
pandas
numpy
nltk
```
## **Deployment**
1. You can deploy this project using:
2. Streamlit Community Cloud
3. GitHub Repository
4. No FastAPI is required.

## **Acknowledgements**

Scikit-learn documentation

Streamlit community

## **Contact**
```
Author: Suryamoorthy
Email: suryamoorthysuresh45@gmail.com
```

