import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

st.set_page_config(page_title="Churn Predictor | Garv Bedi", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
.prediction-box-churn { background: linear-gradient(135deg,#3d0000,#1a0000); border: 2px solid #ff4444; border-radius: 16px; padding: 28px; text-align: center; }
.prediction-box-stay { background: linear-gradient(135deg,#003d1a,#001a0d); border: 2px solid #00cc66; border-radius: 16px; padding: 28px; text-align: center; }
div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model         = joblib.load("model.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    shap_values   = np.load("shap_values.npy")
    X_train       = np.load("X_train.npy")
    X_test        = np.load("X_test.npy")
    y_test        = np.load("y_test.npy")
    y_pred        = np.load("y_pred.npy")
    return model, scaler, feature_names, shap_values, X_train, X_test, y_test, y_pred

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

model, scaler, feature_names, shap_values, X_train, X_test, y_test, y_pred = load_artifacts()
# handle both 2D and 3D shap arrays
shap_values = shap_values if shap_values.ndim == 2 else shap_values[:, :, 1]
df  = load_data()
acc = accuracy_score(y_test, y_pred)

with st.sidebar:
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#00d4ff;'>📉 Churn Predictor</div><div style='font-size:0.75rem;color:#8888aa;'>by Garv Bedi · ML Portfolio</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎛️ Customer Profile")
    gender=st.selectbox("Gender",["Male","Female"]); senior=st.selectbox("Senior Citizen",[0,1])
    partner=st.selectbox("Partner",["Yes","No"]); dependents=st.selectbox("Dependents",["Yes","No"])
    tenure=st.slider("Tenure (months)",0,72,12)
    st.markdown("**📦 Services**")
    phone=st.selectbox("Phone Service",["Yes","No"]); multiple_lines=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    internet=st.selectbox("Internet Service",["Fiber optic","DSL","No"]); online_sec=st.selectbox("Online Security",["Yes","No","No internet service"])
    online_bkp=st.selectbox("Online Backup",["Yes","No","No internet service"]); device_prot=st.selectbox("Device Protection",["Yes","No","No internet service"])
    tech_support=st.selectbox("Tech Support",["Yes","No","No internet service"]); streaming_tv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
    streaming_mv=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
    st.markdown("**💳 Billing**")
    contract=st.selectbox("Contract",["Month-to-month","One year","Two year"]); paperless=st.selectbox("Paperless Billing",["Yes","No"])
    payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    monthly=st.number_input("Monthly Charges ($)",18.0,120.0,65.0,step=0.5); total=st.number_input("Total Charges ($)",0.0,8700.0,float(monthly*tenure),step=1.0)
    predict_btn=st.button("🔮 Predict Churn",use_container_width=True,type="primary")

def encode_input():
    m={"Yes":1,"No":0,"Male":1,"Female":0,"No phone service":2,"No internet service":2,"DSL":0,"Fiber optic":1,"Month-to-month":0,"One year":1,"Two year":2,"Electronic check":0,"Mailed check":1,"Bank transfer (automatic)":2,"Credit card (automatic)":3}
    raw={"gender":m.get(gender,0),"SeniorCitizen":senior,"Partner":m.get(partner,0),"Dependents":m.get(dependents,0),"tenure":tenure,"PhoneService":m.get(phone,0),"MultipleLines":m.get(multiple_lines,0),"InternetService":m.get(internet,0),"OnlineSecurity":m.get(online_sec,0),"OnlineBackup":m.get(online_bkp,0),"DeviceProtection":m.get(device_prot,0),"TechSupport":m.get(tech_support,0),"StreamingTV":m.get(streaming_tv,0),"StreamingMovies":m.get(streaming_mv,0),"Contract":m.get(contract,0),"PaperlessBilling":m.get(paperless,0),"PaymentMethod":m.get(payment,0),"MonthlyCharges":monthly,"TotalCharges":total}
    return pd.DataFrame([raw])[feature_names]

st.markdown("<h1 style='color:#fff;margin-bottom:2px;'>Customer Churn <span style='color:#00d4ff;'>Predictor</span></h1><p style='color:#8888aa;'>Random Forest · SHAP Explainability · Telco Dataset</p>", unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
with c1: st.metric("Accuracy",f"{acc*100:.1f}%")
with c2: st.metric("Dataset Size",f"{len(df):,}")
with c3: st.metric("Churned",f"{df['Churn'].value_counts()['Yes']:,}")
with c4: st.metric("Churn Rate",f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
st.markdown("---")

tab1,tab2,tab3,tab4=st.tabs(["🔮 Prediction","📊 SHAP Explainability","📈 Model Performance","🔍 Data Explorer"])

with tab1:
    st.markdown("### Predict Churn for a New Customer")
    if predict_btn:
        inp=encode_input(); inp_s=scaler.transform(inp); pred=model.predict(inp_s)[0]; proba=model.predict_proba(inp_s)[0]
        ca,cb=st.columns(2)
        with ca:
            if pred==1:
                st.markdown(f"<div class='prediction-box-churn'><div style='font-size:3rem;'>🚨</div><div style='font-size:1.6rem;font-weight:800;color:#ff4444;font-family:Syne,sans-serif;'>WILL CHURN</div><div style='font-size:2.5rem;font-weight:800;color:#ff4444;font-family:Syne,sans-serif;'>{proba[1]*100:.1f}%</div><div style='color:#cc8888;font-size:0.85rem;margin-top:6px;'>Churn Probability</div></div>",unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-box-stay'><div style='font-size:3rem;'>✅</div><div style='font-size:1.6rem;font-weight:800;color:#00cc66;font-family:Syne,sans-serif;'>WILL STAY</div><div style='font-size:2.5rem;font-weight:800;color:#00cc66;font-family:Syne,sans-serif;'>{proba[0]*100:.1f}%</div><div style='color:#88cc88;font-size:0.85rem;margin-top:6px;'>Retention Probability</div></div>",unsafe_allow_html=True)
        with cb:
            fig,ax=plt.subplots(figsize=(5,3),facecolor="#0d0d12"); ax.barh(["Stay","Churn"],[proba[0]*100,proba[1]*100],color=["#00cc66","#ff4444"],height=0.5)
            ax.set_xlim(0,100); ax.set_facecolor("#0d0d12"); ax.tick_params(colors="white"); ax.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
            ax.set_xlabel("Probability (%)",color="white",fontsize=9); ax.set_title("Prediction Confidence",color="white",fontsize=11,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("#### 🧠 Why this prediction? (SHAP Waterfall)")
        exp=shap.TreeExplainer(model); sv=exp(pd.DataFrame(inp_s,columns=feature_names))
        shap.plots.waterfall(sv[0][:,1] if sv.values.ndim==3 else sv[0],max_display=12,show=False)
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close("all")
    else:
        st.info("👈 Configure the customer profile in the sidebar and click **Predict Churn**.")

with tab2:
    st.markdown("### SHAP — Global Feature Importance")
    cs1,cs2=st.columns(2)
    with cs1:
        st.markdown("#### 📊 SHAP Beeswarm")
        shap.summary_plot(shap_values,X_train,feature_names=feature_names,show=False,plot_size=None)
        plt.gcf().set_facecolor("#0d0d12"); plt.tight_layout(); st.pyplot(plt.gcf()); plt.close("all")
    with cs2:
        st.markdown("#### 🏆 Mean |SHAP| Ranking")
        ms=np.abs(shap_values).mean(axis=0); sdf=pd.DataFrame({"Feature":feature_names,"Mean |SHAP|":ms}).sort_values("Mean |SHAP|",ascending=True).tail(12)
        fig4,ax4=plt.subplots(figsize=(7,6),facecolor="#0d0d12"); ax4.set_facecolor("#0d0d12")
        ax4.barh(sdf["Feature"],sdf["Mean |SHAP|"],color=["#00d4ff" if v>sdf["Mean |SHAP|"].median() else "#2a5580" for v in sdf["Mean |SHAP|"]])
        ax4.tick_params(colors="white",labelsize=9); ax4.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax4.set_xlabel("Mean |SHAP Value|",color="white"); ax4.set_title("Top Features",color="white",fontweight="bold")
        plt.tight_layout(); st.pyplot(fig4); plt.close()
    dep_feat=st.selectbox("Select feature for dependence plot:",feature_names,index=feature_names.index("tenure") if "tenure" in feature_names else 0)
    fig5,ax5=plt.subplots(figsize=(10,4),facecolor="#0d0d12")
    shap.dependence_plot(dep_feat,shap_values,X_train,feature_names=feature_names,ax=ax5,show=False)
    fig5.set_facecolor("#0d0d12"); ax5.set_facecolor("#0d0d12"); ax5.tick_params(colors="white"); ax5.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
    ax5.set_title(f"SHAP Dependence: {dep_feat}",color="white",fontweight="bold"); ax5.xaxis.label.set_color("white"); ax5.yaxis.label.set_color("white")
    plt.tight_layout(); st.pyplot(fig5); plt.close()

with tab3:
    st.markdown("### Model Performance")
    cm1,cm2=st.columns(2)
    with cm1:
        st.markdown("#### Confusion Matrix")
        cm=confusion_matrix(y_test,y_pred); fig6,ax6=plt.subplots(figsize=(5,4),facecolor="#0d0d12")
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Stay","Churn"],yticklabels=["Stay","Churn"],ax=ax6,cbar=False,annot_kws={"size":14,"weight":"bold","color":"white"})
        ax6.set_facecolor("#0d0d12"); fig6.set_facecolor("#0d0d12"); ax6.tick_params(colors="white"); ax6.set_xlabel("Predicted",color="white"); ax6.set_ylabel("Actual",color="white"); ax6.set_title("Confusion Matrix",color="white",fontweight="bold")
        plt.tight_layout(); st.pyplot(fig6); plt.close()
    with cm2:
        st.markdown("#### ROC Curve")
        fpr,tpr,_=roc_curve(y_test,model.predict_proba(X_test)[:,1]); ra=auc(fpr,tpr)
        fig7,ax7=plt.subplots(figsize=(5,4),facecolor="#0d0d12"); ax7.set_facecolor("#0d0d12")
        ax7.plot(fpr,tpr,color="#00d4ff",lw=2,label=f"AUC={ra:.3f}"); ax7.plot([0,1],[0,1],color="#444",linestyle="--",lw=1); ax7.fill_between(fpr,tpr,alpha=0.1,color="#00d4ff")
        ax7.tick_params(colors="white"); ax7.spines[["top","right","bottom","left"]].set_color("#2a2a4a"); ax7.set_xlabel("FPR",color="white"); ax7.set_ylabel("TPR",color="white"); ax7.set_title("ROC Curve",color="white",fontweight="bold"); ax7.legend(facecolor="#1a1a2e",labelcolor="white")
        plt.tight_layout(); st.pyplot(fig7); plt.close()
    st.markdown("#### Classification Report")
    rdf=pd.DataFrame(classification_report(y_test,y_pred,target_names=["Stay","Churn"],output_dict=True)).T.round(3)
    st.dataframe(rdf.style.background_gradient(cmap="Blues",subset=["precision","recall","f1-score"]),use_container_width=True)

with tab4:
    st.markdown("### Data Explorer")
    dd1,dd2=st.columns(2)
    with dd1:
        st.markdown("#### Churn Distribution"); counts=df["Churn"].value_counts()
        fig8,ax8=plt.subplots(figsize=(5,4),facecolor="#0d0d12"); ax8.set_facecolor("#0d0d12")
        ax8.bar(counts.index,counts.values,color=["#00d4ff","#ff4444"],width=0.5); ax8.tick_params(colors="white"); ax8.spines[["top","right","bottom","left"]].set_color("#2a2a4a")
        ax8.set_ylabel("Count",color="white"); ax8.set_title("Churn vs Retained",color="white",fontweight="bold")
        [ax8.text(i,v+20,str(v),ha="center",color="white",fontweight="bold") for i,v in enumerate(counts.values)]
        plt.tight_layout(); st.pyplot(fig8); plt.close()
    with dd2:
        st.markdown("#### Contract vs Churn")
        fig9,ax9=plt.subplots(figsize=(5,4),facecolor="#0d0d12"); ax9.set_facecolor("#0d0d12")
        pd.crosstab(df["Contract"],df["Churn"]).plot(kind="bar",ax=ax9,color=["#00d4ff","#ff4444"],edgecolor="none",rot=30)
        ax9.tick_params(colors="white"); ax9.spines[["top","right","bottom","left"]].set_color("#2a2a4a"); ax9.set_ylabel("Count",color="white"); ax9.set_title("Contract Type vs Churn",color="white",fontweight="bold"); ax9.legend(["Stay","Churn"],facecolor="#1a1a2e",labelcolor="white")
        plt.tight_layout(); st.pyplot(fig9); plt.close()
    fig10,ax10=plt.subplots(figsize=(10,3.5),facecolor="#0d0d12"); ax10.set_facecolor("#0d0d12")
    [ax10.hist(df[df["Churn"]==l]["MonthlyCharges"],bins=30,alpha=0.6,color=c,label=f"Churn={l}",edgecolor="none") for l,c in [("No","#00d4ff"),("Yes","#ff4444")]]
    ax10.tick_params(colors="white"); ax10.spines[["top","right","bottom","left"]].set_color("#2a2a4a"); ax10.set_xlabel("Monthly Charges ($)",color="white"); ax10.set_ylabel("Count",color="white"); ax10.set_title("Monthly Charges Distribution",color="white",fontweight="bold"); ax10.legend(facecolor="#1a1a2e",labelcolor="white")
    plt.tight_layout(); st.pyplot(fig10); plt.close()
    with st.expander("📋 Raw Dataset"): st.dataframe(df.head(200),use_container_width=True)
