import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import shap
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Garv Bedi",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.main { background-color: #0d0d12; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 4px;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #00d4ff; font-family: 'Syne', sans-serif; }
.metric-label { font-size: 0.8rem; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }

.prediction-box-churn {
    background: linear-gradient(135deg, #3d0000, #1a0000);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.prediction-box-stay {
    background: linear-gradient(135deg, #003d1a, #001a0d);
    border: 2px solid #00cc66;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-emoji { font-size: 3rem; }
.pred-label { font-size: 1.6rem; font-weight: 800; font-family: 'Syne', sans-serif; }

.sidebar-header {
    background: linear-gradient(135deg, #0d0d12, #1a1a2e);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid #2a2a4a;
}

.tab-content { padding: 10px 0; }

div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/garvbedi/Customer-Churn-Prediction/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    return df

@st.cache_resource
def train_model(df):
    df_enc = df.copy()
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest (better for SHAP TreeExplainer)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, X, y, X_train, X_test, y_train, y_test, y_pred, acc, df_enc

@st.cache_resource
def get_shap_values(_model, _X_train):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X_train)
    return explainer, shap_values

# ─── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading data & training model..."):
    df = load_data()
    model, scaler, X, y, X_train, X_test, y_train, y_test, y_pred, acc, df_enc = train_model(df)
    explainer, shap_values = get_shap_values(model, X_train)

feature_names = X.columns.tolist()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-header'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#00d4ff;'>📉 Churn Predictor</div>
        <div style='font-size:0.75rem;color:#8888aa;margin-top:4px;'>by Garv Bedi · ML Portfolio</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Customer Profile")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown("**📦 Services**")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("**💳 Billing**")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total = st.number_input("Total Charges ($)", 0.0, 8700.0, monthly * tenure, step=1.0)

    predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ─── Input Encoding ───────────────────────────────────────────────────────────
def encode_input():
    mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
               "No phone service": 2, "No internet service": 2,
               "DSL": 0, "Fiber optic": 1, "No": 0,
               "Month-to-month": 0, "One year": 1, "Two year": 2,
               "Electronic check": 0, "Mailed check": 1,
               "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}

    raw = {
        "gender": mapping.get(gender, 0),
        "SeniorCitizen": senior,
        "Partner": mapping.get(partner, 0),
        "Dependents": mapping.get(dependents, 0),
        "tenure": tenure,
        "PhoneService": mapping.get(phone, 0),
        "MultipleLines": mapping.get(multiple_lines, 0),
        "InternetService": mapping.get(internet, 0),
        "OnlineSecurity": mapping.get(online_sec, 0),
        "OnlineBackup": mapping.get(online_bkp, 0),
        "DeviceProtection": mapping.get(device_prot, 0),
        "TechSupport": mapping.get(tech_support, 0),
        "StreamingTV": mapping.get(streaming_tv, 0),
        "StreamingMovies": mapping.get(streaming_mv, 0),
        "Contract": mapping.get(contract, 0),
        "PaperlessBilling": mapping.get(paperless, 0),
        "PaymentMethod": mapping.get(payment, 0),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    return pd.DataFrame([raw])[feature_names]

# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#ffffff;margin-bottom:2px;'>Customer Churn <span style='color:#00d4ff;'>Predictor</span></h1>
<p style='color:#8888aa;font-size:1rem;margin-bottom:20px;'>Random Forest · SHAP Explainability · Telco Dataset</p>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
churn_rate = df["Churn"].value_counts(normalize=True)["Yes"] * 100
total_cust = len(df)
churned = df["Churn"].value_counts()["Yes"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", f"{acc*100:.1f}%")
with col2:
    st.metric("Dataset Size", f"{total_cust:,}")
with col3:
    st.metric("Churned Customers", f"{churned:,}")
with col4:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 SHAP Explainability", "📈 Model Performance", "🔍 Data Explorer"])

# ══════════════════ TAB 1: PREDICTION ══════════════════
with tab1:
    st.markdown("### Predict Churn for a New Customer")
    st.caption("Adjust the customer profile in the sidebar, then click **Predict Churn**.")

    if predict_btn:
        input_df = encode_input()
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        churn_prob = proba[1] * 100
        stay_prob = proba[0] * 100

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if pred == 1:
                st.markdown(f"""
                <div class='prediction-box-churn'>
                    <div class='pred-emoji'>🚨</div>
                    <div class='pred-label' style='color:#ff4444;'>WILL CHURN</div>
                    <div style='font-size:2.5rem;font-weight:800;color:#ff4444;font-family:Syne,sans-serif;'>{churn_prob:.1f}%</div>
                    <div style='color:#cc8888;font-size:0.85rem;margin-top:6px;'>Churn Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='prediction-box-stay'>
                    <div class='pred-emoji'>✅</div>
                    <div class='pred-label' style='color:#00cc66;'>WILL STAY</div>
                    <div style='font-size:2.5rem;font-weight:800;color:#00cc66;font-family:Syne,sans-serif;'>{stay_prob:.1f}%</div>
                    <div style='color:#88cc88;font-size:0.85rem;margin-top:6px;'>Retention Probability</div>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0d0d12")
            colors = ["#00cc66", "#ff4444"]
            bars = ax.barh(["Stay", "Churn"], [stay_prob, churn_prob], color=colors, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_facecolor("#0d0d12")
            ax.tick_params(colors="white")
            ax.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
            ax.set_xlabel("Probability (%)", color="white", fontsize=9)
            ax.set_title("Prediction Confidence", color="white", fontsize=11, fontweight="bold")
            for bar, val in zip(bars, [stay_prob, churn_prob]):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                        va="center", color="white", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # SHAP Waterfall for this prediction
        st.markdown("#### 🧠 Why this prediction? (SHAP Waterfall)")
        shap_explainer_single = shap.TreeExplainer(model)
        sv = shap_explainer_single(pd.DataFrame(input_scaled, columns=feature_names))
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0d0d12")
        shap.plots.waterfall(sv[0][:, 1] if sv.values.ndim == 3 else sv[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close("all")

    else:
        st.info("👈 Configure the customer profile in the sidebar and click **Predict Churn** to see results.")

# ══════════════════ TAB 2: SHAP ══════════════════
with tab2:
    st.markdown("### SHAP — Global Feature Importance")
    st.caption("SHAP (SHapley Additive exPlanations) explains *which features* drive churn predictions across all customers.")

    shap_class = shap_values[1] if isinstance(shap_values, list) else shap_values

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### 📊 SHAP Summary (Beeswarm)")
        fig3, ax3 = plt.subplots(figsize=(7, 6), facecolor="#0d0d12")
        shap.summary_plot(shap_class, X_train, feature_names=feature_names, show=False, plot_size=None)
        plt.gcf().set_facecolor("#0d0d12")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close("all")

    with col_s2:
        st.markdown("#### 🏆 Mean |SHAP| — Feature Ranking")
        mean_shap = np.abs(shap_class).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_shap})
        shap_df = shap_df.sort_values("Mean |SHAP|", ascending=True).tail(12)

        fig4, ax4 = plt.subplots(figsize=(7, 6), facecolor="#0d0d12")
        ax4.set_facecolor("#0d0d12")
        colors_bar = ["#00d4ff" if v > shap_df["Mean |SHAP|"].median() else "#2a5580"
                      for v in shap_df["Mean |SHAP|"]]
        ax4.barh(shap_df["Feature"], shap_df["Mean |SHAP|"], color=colors_bar)
        ax4.tick_params(colors="white", labelsize=9)
        ax4.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax4.set_xlabel("Mean |SHAP Value|", color="white")
        ax4.set_title("Top Features by SHAP Importance", color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("#### 🔗 SHAP Dependence Plot")
    dep_feat = st.selectbox("Select feature to explore:", feature_names, index=feature_names.index("tenure"))
    fig5, ax5 = plt.subplots(figsize=(10, 4), facecolor="#0d0d12")
    shap.dependence_plot(dep_feat, shap_class, X_train,
                         feature_names=feature_names, ax=ax5, show=False)
    fig5.set_facecolor("#0d0d12")
    ax5.set_facecolor("#0d0d12")
    ax5.tick_params(colors="white")
    ax5.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
    ax5.set_title(f"SHAP Dependence: {dep_feat}", color="white", fontweight="bold")
    ax5.xaxis.label.set_color("white")
    ax5.yaxis.label.set_color("white")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ══════════════════ TAB 3: MODEL PERFORMANCE ══════════════════
with tab3:
    st.markdown("### Model Performance")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig6, ax6 = plt.subplots(figsize=(5, 4), facecolor="#0d0d12")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
                    ax=ax6, cbar=False,
                    annot_kws={"size": 14, "weight": "bold", "color": "white"})
        ax6.set_facecolor("#0d0d12")
        fig6.set_facecolor("#0d0d12")
        ax6.tick_params(colors="white")
        ax6.set_xlabel("Predicted", color="white")
        ax6.set_ylabel("Actual", color="white")
        ax6.set_title("Confusion Matrix", color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    with col_m2:
        st.markdown("#### ROC Curve")
        proba_test = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        roc_auc = auc(fpr, tpr)
        fig7, ax7 = plt.subplots(figsize=(5, 4), facecolor="#0d0d12")
        ax7.set_facecolor("#0d0d12")
        ax7.plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax7.plot([0, 1], [0, 1], color="#444", linestyle="--", lw=1)
        ax7.fill_between(fpr, tpr, alpha=0.1, color="#00d4ff")
        ax7.tick_params(colors="white")
        ax7.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax7.set_xlabel("False Positive Rate", color="white")
        ax7.set_ylabel("True Positive Rate", color="white")
        ax7.set_title("ROC Curve", color="white", fontweight="bold")
        ax7.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Stay", "Churn"], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                 use_container_width=True)

# ══════════════════ TAB 4: DATA EXPLORER ══════════════════
with tab4:
    st.markdown("### Data Explorer")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Churn Distribution")
        fig8, ax8 = plt.subplots(figsize=(5, 4), facecolor="#0d0d12")
        ax8.set_facecolor("#0d0d12")
        counts = df["Churn"].value_counts()
        ax8.bar(counts.index, counts.values, color=["#00d4ff","#ff4444"], width=0.5)
        ax8.tick_params(colors="white")
        ax8.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax8.set_ylabel("Count", color="white")
        ax8.set_title("Churn vs Retained", color="white", fontweight="bold")
        for i, v in enumerate(counts.values):
            ax8.text(i, v + 20, str(v), ha="center", color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()

    with col_d2:
        st.markdown("#### Contract Type vs Churn")
        fig9, ax9 = plt.subplots(figsize=(5, 4), facecolor="#0d0d12")
        ax9.set_facecolor("#0d0d12")
        ct = pd.crosstab(df["Contract"], df["Churn"])
        ct.plot(kind="bar", ax=ax9, color=["#00d4ff","#ff4444"], edgecolor="none", rot=30)
        ax9.tick_params(colors="white")
        ax9.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax9.set_xlabel("", color="white")
        ax9.set_ylabel("Count", color="white")
        ax9.set_title("Contract Type vs Churn", color="white", fontweight="bold")
        ax9.legend(["Stay", "Churn"], facecolor="#1a1a2e", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig9)
        plt.close()

    st.markdown("#### Monthly Charges Distribution by Churn")
    fig10, ax10 = plt.subplots(figsize=(10, 3.5), facecolor="#0d0d12")
    ax10.set_facecolor("#0d0d12")
    for label, color in [("No", "#00d4ff"), ("Yes", "#ff4444")]:
        vals = df[df["Churn"] == label]["MonthlyCharges"]
        ax10.hist(vals, bins=30, alpha=0.6, color=color, label=f"Churn={label}", edgecolor="none")
    ax10.tick_params(colors="white")
    ax10.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
    ax10.set_xlabel("Monthly Charges ($)", color="white")
    ax10.set_ylabel("Count", color="white")
    ax10.set_title("Monthly Charges Distribution", color="white", fontweight="bold")
    ax10.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close()

    with st.expander("📋 Show Raw Dataset"):
        st.dataframe(df.head(200), use_container_width=True)
