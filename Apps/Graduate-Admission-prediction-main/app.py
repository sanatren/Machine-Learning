import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Graduate Admission Prediction",
    page_icon=":mortar_board:",  # Streamlit emoji syntax for 🎓
    layout="wide"
)

# Hide default Streamlit features (menu, footer)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for aesthetics
custom_css = """
<style>
body {
    background: linear-gradient(to right, #f9f9f9, #e6f7ff);
    font-family: 'Arial', sans-serif;
}
.sidebar .sidebar-content {
    background: #f0f0f5;
}
.big-title {
    font-size: 3em;
    font-weight: 700;
    text-align: center;
    color: #003366;
    padding: 20px 0;
}
.subheader {
    font-size: 1.5em;
    font-weight: 600;
    color: #004080;
    padding: 10px 0;
}
.section {
    background: #ffffffcc;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}
hr {
    border: none;
    height: 1px;
    background: #ccc;
    margin: 20px 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    Admission = pd.read_csv('Admission_Predict_Ver1.1.csv')
    Admission = Admission.drop(columns='Serial No.')
    Admission = Admission.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'})
    return Admission

Admission = load_data()

# Split the data
X = Admission.drop(columns='Chance of Admit')
Y = Admission['Chance of Admit']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Load trained model and scaler
with open('bestgrad.pkl', 'rb') as model_file:
    reg = pickle.load(model_file)

with open('minmax.pkl', 'rb') as scaler_file:
    minmax = pickle.load(scaler_file)

# Prepare navigation
st.sidebar.title("Navigation")
nav = st.sidebar.radio("", ["Home", "EDA", "Model Evaluation", "Predict"])

st.sidebar.write("---")
st.sidebar.write("**Created by Sanatan Khemariya**")

if nav == "Home":
    st.markdown("<h1 class='big-title'>Graduate Admission Predictor</h1>", unsafe_allow_html=True)
    st.write("""
    This application predicts the chance of admission for a student applying to graduate school based on their academic and profile features such as GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research experience.
    """)
    st.write("---")
    st.write("**What can you find here?**")
    st.write("- **EDA:** Explore the data, distributions, correlations, and relationships between features.")
    st.write("- **Model Evaluation:** Review the performance of various regression models tested and see how the best model was chosen.")
    st.write("- **Predict:** Use the best model to predict the chance of admission for a new applicant.")

elif nav == "EDA":
    st.markdown("<h2 class='subheader'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.write("Below are some of the EDA plots that were generated:")

    fig_corr, ax = plt.subplots(figsize=(12, 10))
    corr = Admission.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, linewidths=1.5, cmap="YlGnBu", annot_kws={"size": 8}, ax=ax)
    st.write("### Correlation Heatmap")
    st.pyplot(fig_corr)

    def plot_reg(x, y, title):
        fig, ax = plt.subplots()
        sns.regplot(x=x, y=y, data=Admission, ax=ax)
        ax.set_title(title)
        return fig

    st.write("### GRE Score vs TOEFL Score")
    st.pyplot(plot_reg("GRE Score", "TOEFL Score", "GRE Score vs TOEFL Score"))

    st.write("### GRE Score vs CGPA")
    st.pyplot(plot_reg("GRE Score", "CGPA", "GRE Score vs CGPA"))

    st.write("### LOR vs CGPA")
    st.pyplot(plot_reg("LOR", "CGPA", "LOR vs CGPA"))

    st.write("### CGPA vs SOP")
    st.pyplot(plot_reg("CGPA", "SOP", "CGPA vs SOP"))

    st.write("### GRE Score vs SOP")
    st.pyplot(plot_reg("GRE Score", "SOP", "GRE Score vs SOP"))

    st.write("### TOEFL Score vs SOP")
    st.pyplot(plot_reg("TOEFL Score", "SOP", "TOEFL Score vs SOP"))

    st.write("---")
    st.write("**Observations:**")
    st.write("- GRE Score and TOEFL Score appear positively correlated.")
    st.write("- CGPA correlates strongly with the Chance of Admission.")
    st.write("- Research experience also influences the chance of admission.")

elif nav == "Model Evaluation":
    st.markdown("<h2 class='subheader'>Model Evaluation Results</h2>", unsafe_allow_html=True)
    st.write("We tried multiple regression models and evaluated them using R² and RMSE.")

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Lasso, BayesianRidge, ElasticNet, HuberRegressor
    from xgboost import XGBRegressor

    models = [
        ('Decision Tree', DecisionTreeRegressor()),
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor()),
        ('KNeighbors', KNeighborsRegressor(n_neighbors=2)),
        ('SVM', SVR()),
        ('AdaBoost', AdaBoostRegressor()),
        ('GradientBoosting', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('BayesianRidge', BayesianRidge()),
        ('ElasticNet', ElasticNet()),
        ('HuberRegressor', HuberRegressor())
    ]

    x_train_scaled = minmax.fit_transform(x_train)  # Fit on train
    x_test_scaled = minmax.transform(x_test)

    results_data = []
    for name, model in models:
        try:
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            results_data.append([name, rmse, r2])
        except Exception as e:
            st.warning(f"Model {name} failed to run: {e}")

    results_df = pd.DataFrame(results_data, columns=["Model", "RMSE", "R² Score"])
    st.write("### Model Comparison")
    st.dataframe(results_df)

    best_model = results_df.iloc[results_df['R² Score'].idxmax()]
    st.write(f"**Best Model:** {best_model['Model']} with R² = {best_model['R² Score']:.4f}")

    st.write("### Feature Importances (Random Forest)")
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train_scaled, y_train)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20)
    fig_rf, ax = plt.subplots(figsize=(12, 8))
    feat_imp.plot(kind='barh', ax=ax)
    ax.set_title("Feature Importances (Random Forest)")
    st.pyplot(fig_rf)

    # Linear Regression Residual Analysis
    st.write("### Residual Analysis for Linear Regression")
    y_pred_reg = reg.predict(x_test_scaled)
    residuals = y_test - y_pred_reg

    fig_res1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred_reg, alpha=0.7)
    ax1.set_title("Predicted vs Actual")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")

    fig_res2, ax2 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title("Residuals Distribution")

    fig_res3, ax3 = plt.subplots()
    ax3.scatter(y_pred_reg, residuals, alpha=0.7)
    ax3.set_title("Residuals vs Predictions")
    ax3.set_xlabel("Predictions")
    ax3.set_ylabel("Residuals")

    st.pyplot(fig_res1)
    st.pyplot(fig_res2)
    st.pyplot(fig_res3)

    # OLS summary
    st.write("### OLS Summary")
    model_ols = sm.OLS(y_train, x_train_scaled).fit()
    st.text(model_ols.summary())

elif nav == "Predict":
    st.markdown("<h2 class='subheader'>Predict Your Chance of Admission</h2>", unsafe_allow_html=True)

    gre = st.slider("GRE Score", 0, 340, 320)
    toefl = st.slider("TOEFL Score", 0, 120, 110)
    rating = st.slider("University Rating", 1, 5, 3)
    sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, 0.5)
    lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, 0.5)
    cgpa = st.slider("CGPA", 0.0, 10.0, 8.5, 0.1)
    research = st.radio("Research Experience", [0, 1])

    input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    scaled_input = minmax.transform(input_data)
    prediction = reg.predict(scaled_input)

    if st.button("Predict"):
        st.write(f"**Predicted Chance of Admission:** {prediction[0]:.2f}")
        st.write("This value is on a scale of 0 to 1, representing the probability of admission.")
