# 📰 Fake News Detector

A machine learning web app that classifies news articles as **Real** or **Fake** using TF-IDF vectorization and Logistic Regression, deployed with Streamlit.

## 🚀 Live Demo
[Add your Streamlit link here once deployed]

## 🛠️ Tech Stack
- Python, Pandas
- Scikit-learn (TF-IDF, Logistic Regression)
- Streamlit (web app + deployment)

## 📊 Dataset
[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## ⚙️ How it Works
1. News articles are cleaned (lowercased, links/punctuation removed)
2. Text is converted into numerical features using TF-IDF
3. A Logistic Regression model is trained to classify Real vs Fake news
4. A Streamlit web app lets users paste any news text and get an instant prediction with confidence score

## 💻 Run Locally
```bash
git clone https://github.com/Manish1Gupta/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
python train_model.py     # trains and saves the model
streamlit run app.py      # launches the web app
```

## 📈 Results
Achieved ~95% accuracy on the held-out test set.

## 👤 Author
**Manish Gupta**
[GitHub](https://github.com/Manish1Gupta)
