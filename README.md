# 📉 Customer Churn Prediction — with SHAP Explainability

> **Live App →** [your-streamlit-link-here]  
> Built by [Garv Bedi](https://github.com/garvbedi) · ML Portfolio Project

---

## 🔍 What This Project Does

Predicts whether a telecom customer will churn (cancel their subscription) using a **Random Forest** classifier trained on the IBM Telco Churn dataset.

Beyond just predicting — it **explains *why*** using SHAP (SHapley Additive exPlanations), making the model interpretable and production-ready.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Live Prediction** | Input any customer's profile → get churn probability instantly |
| 🧠 **SHAP Waterfall** | Per-prediction explanation of which features pushed the result |
| 📊 **SHAP Beeswarm** | Global feature importance across the entire dataset |
| 🔗 **Dependence Plots** | How individual features influence churn probability |
| 📈 **ROC Curve** | Model performance with AUC score |
| 🔍 **Data Explorer** | EDA charts on the raw Telco dataset |

---

## 🛠️ Tech Stack

- **Python** — core language
- **Scikit-learn** — Random Forest classifier
- **SHAP** — model explainability
- **Streamlit** — interactive web app
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualizations

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── app.py                          # Streamlit app (main)
├── requirements.txt                # Dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── Churn_Prediction.ipynb          # Original EDA notebook
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/garvbedi/Customer-Churn-Prediction
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Results

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| AUC-ROC | ~0.85+ |
| Explainability | ✅ SHAP |

---

## 🧠 What is SHAP?

SHAP assigns each feature a "Shapley value" — the contribution of that feature to the final prediction for a specific customer. Unlike feature importance, SHAP is:
- **Local**: explains each prediction individually
- **Consistent**: guaranteed to be mathematically fair
- **Directional**: shows whether a feature pushed toward churn or retention

---

## 🚀 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select `app.py`
4. Deploy!
