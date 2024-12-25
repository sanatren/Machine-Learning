import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
from bs4 import BeautifulSoup
import feedparser

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import plotly.express as px

# --------------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------------
st.set_page_config(
    page_title="Smart Flight Price Predictor",
    layout="wide"
)

# Set a Seaborn style for older plots (optional)
sns.set_style("whitegrid")

# --------------------------------------------------------
# Custom CSS
# --------------------------------------------------------
st.markdown(
    """
    <style>
    .main { padding: 1rem 2rem; background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    h1 { color: #2c3e50; text-align: center; animation: fadeIn 1s ease-in; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Airlines News Scraping Function
# --------------------------------------------------------

def fetch_airline_news():
    feed_url = "https://simpleflying.com/feed/category/aviation-news/airlines/"
    try:
        feed = feedparser.parse(feed_url)
        news_items = []
        for entry in feed.entries[:5]:  # Fetch the latest 5 news articles
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published if 'published' in entry else "Unknown"
            })
        return news_items
    except Exception as e:
        return f"Error fetching news: {str(e)}"
# --------------------------------------------------------
# 1. Data Loading & Preprocessing
# --------------------------------------------------------
@st.cache_data
def load_raw_data():
    """Loads the raw training data (Excel) and returns it unaltered."""
    df_train = pd.read_excel('dataset/Data_Train.xlsx')
    return df_train


def preprocess_data(df):
    """Apply the same transformations as in your Jupyter Notebook
       so that df can be used for analysis and modeling.
    """
    df = df.copy()
    df.dropna(how='all', inplace=True)  # Just in case any completely empty rows

    # Fill missing Price with median if needed
    if 'Price' in df.columns:
        df['Price'].fillna(df['Price'].median(), inplace=True)

    # --- Extract Date, Month, Year ---
    df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
    df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
    df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)
    df.drop(columns='Date_of_Journey', inplace=True, errors='ignore')

    # --- Dep_Time ---
    df['Dep_Time_hrs'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dep_Time_Mns'] = df['Dep_Time'].str.split(':').str[1].astype(int)
    df.drop(columns='Dep_Time', inplace=True, errors='ignore')

    # --- Arrival_Time ---
    df['Arrival_Time_hours'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
    df['Arrival_Time_Minutes'] = (
        df['Arrival_Time'].str.split(':').str[1].str.split(' ').str[0].astype(int)
    )
    df.drop(columns='Arrival_Time', inplace=True, errors='ignore')

    # --- Duration ---
    duration_split = df['Duration'].str.extract(r'(?:(\d+)h)? ?(?:(\d+)m)?')
    df['Duration_hrs'] = duration_split[0].fillna(0).astype(int)
    df['Duration_Mins'] = duration_split[1].fillna(0).astype(int)
    df.drop(columns='Duration', inplace=True, errors='ignore')

    # --- Total_Stops ---
    df['Total_Stops'] = df['Total_Stops'].map({
        'non-stop': 0, '1 stop': 1, '2 stop': 2, '3 stop': 3, '4 stop': 4
    })
    df['Total_Stops'].fillna(1, inplace=True)
    df['Total_Stops'] = df['Total_Stops'].astype(int)

    # --- Drop columns not needed ---
    for col in ['Route', 'Additional_Info']:
        if col in df.columns:
            df.drop(columns=col, inplace=True, errors='ignore')

    # Copy for analysis (keeps Airline, Source, Destination as is)
    df_analysis = df.copy()

    # One-hot encode for modeling
    if 'Airline' in df.columns:
        airline_dummies = pd.get_dummies(df[['Airline']], drop_first=True)
        source_dummies = pd.get_dummies(df[['Source']], drop_first=True)
        dest_dummies = pd.get_dummies(df[['Destination']], drop_first=True)

        df = pd.concat([df, airline_dummies, source_dummies, dest_dummies], axis=1)

        for col in ['Airline', 'Source', 'Destination']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

    return df_analysis, df


@st.cache_data
def load_training_data():
    """Loads & preprocesses the training data, returning both:
       - df_analysis (for visual EDA with string columns)
       - df_model (fully numeric, ready for modeling).
    """
    raw_df = load_raw_data()
    df_analysis, df_model = preprocess_data(raw_df)
    return df_analysis, df_model


# --------------------------------------------------------
# 2. Plotly Dark Themed Boxplots
# --------------------------------------------------------
def create_dark_boxplot_by_airline(df):
    """Boxplot of Price by Airline with Plotly Dark Theme."""
    # Must have 'Airline' and 'Price' columns in df
    if 'Airline' not in df.columns or 'Price' not in df.columns:
        return None
    fig = px.box(
        df,
        x="Airline",
        y="Price",
        template="plotly_dark",
        color_discrete_sequence=["#636EFA"],  # A nice blue
        labels={"Airline": "Airline", "Price": "Price (INR)"}
    )
    fig.update_layout(
        title="Price Distribution by Airline",
        xaxis_title=None,
        yaxis_title="Price",
        font=dict(size=14),
    )
    return fig


def create_dark_boxplot_by_source(df):
    """Boxplot of Price by Source with Plotly Dark Theme."""
    if 'Source' not in df.columns or 'Price' not in df.columns:
        return None
    fig = px.box(
        df,
        x="Source",
        y="Price",
        template="plotly_dark",
        color_discrete_sequence=["#636EFA"],
        labels={"Source": "Source", "Price": "Price (INR)"}
    )
    fig.update_layout(
        title="Price Distribution by Source",
        xaxis_title=None,
        yaxis_title="Price",
        font=dict(size=14),
    )
    return fig


def create_dark_boxplot_by_stops(df):
    """Boxplot of Price by Number of Stops with Plotly Dark Theme."""
    # Must have 'Total_Stops' and 'Price' columns in df
    if 'Total_Stops' not in df.columns or 'Price' not in df.columns:
        return None
    fig = px.box(
        df,
        x="Total_Stops",
        y="Price",
        template="plotly_dark",
        color_discrete_sequence=["#636EFA"],
        labels={"Total_Stops": "Total Stops", "Price": "Price (INR)"}
    )
    fig.update_layout(
        title="Price Distribution by Number of Stops",
        xaxis_title=None,
        yaxis_title="Price",
        font=dict(size=14),
    )
    return fig


# --------------------------------------------------------
# 3. Additional Seaborn/Matplotlib EDA (Optional)
# --------------------------------------------------------
def create_price_distribution_plot(df_analysis):
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df_analysis, x='Price', bins=50, kde=True, color='steelblue')
    plt.title('Flight Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    return fig


def create_stops_analysis_plot(df_analysis):
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_analysis, x='Total_Stops', y='Price', palette='Blues')
    plt.title('Price vs Number of Stops')
    return fig


def create_airline_price_plot(df_analysis):
    if 'Airline' not in df_analysis.columns:
        return None
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_analysis, x='Airline', y='Price', palette='Blues')
    plt.xticks(rotation=45, ha='right')
    plt.title('Price Distribution by Airline')
    return fig


def create_correlation_heatmap(df_model):
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    correlation = df_model[numeric_cols].corr()
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0)
    plt.title('Feature Correlation Heatmap')
    return fig


def create_feature_importance_plot(df_model):
    if 'Price' not in df_model.columns:
        return None
    X = df_model.drop('Price', axis=1)
    y = df_model['Price']
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    fig = plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'][-10:], importance['importance'][-10:], color='steelblue')
    plt.title('Top 10 Most Important Features')
    return fig


# --------------------------------------------------------
# 4. Model Training and Evaluation
# --------------------------------------------------------
def train_baseline_model(df_model):
    X = df_model.drop('Price', axis=1)
    y = df_model['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def show_model_evaluation(model, X_test, y_test, title="Model Performance"):
    st.header(title)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"₹{rmse:,.2f}")
    col3.metric("MAE", f"₹{mae:,.2f}")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Prediction vs Actual Prices')
    st.pyplot(fig)


# --------------------------------------------------------
# 5. Preprocessing User Input for Prediction
# --------------------------------------------------------
def preprocess_input_for_inference(data):
    """Preprocess a single user input dictionary to match model features."""
    try:
        total_stops_map = {'non-stop': 0, '1 stop': 1, '2 stop': 2, '3 stop': 3, '4 stop': 4}
        total_stops_val = total_stops_map[data['Total_Stops']]

        # Date, Month, Year
        date_str = data['Date_of_Journey']
        dd, mm, yyyy = date_str.split('/')
        dd, mm, yyyy = int(dd), int(mm), int(yyyy)

        # Depart time
        dep_hr, dep_mn = data['Dep_Time'].split(':')
        dep_hr, dep_mn = int(dep_hr), int(dep_mn)

        # Arrival time
        arr_hr, arr_mn = data['Arrival_Time'].split(':')
        arr_hr, arr_mn = int(arr_hr), int(arr_mn)

        # Duration
        duration_parts = data['Duration'].split()
        dur_hrs, dur_mins = 0, 0
        if len(duration_parts) >= 1 and 'h' in duration_parts[0]:
            dur_hrs = int(duration_parts[0].replace('h', ''))
        if len(duration_parts) >= 2 and 'm' in duration_parts[1]:
            dur_mins = int(duration_parts[1].replace('m', ''))

        # Build a minimal DataFrame with 1 row
        input_df = pd.DataFrame({
            'Total_Stops': [total_stops_val],
            'Date': [dd],
            'Month': [mm],
            'Year': [yyyy],
            'Dep_Time_hrs': [dep_hr],
            'Dep_Time_Mns': [dep_mn],
            'Arrival_Time_hours': [arr_hr],
            'Arrival_Time_Minutes': [arr_mn],
            'Duration_hrs': [dur_hrs],
            'Duration_Mins': [dur_mins],
            'Airline': [data['Airline']],
            'Source': [data['Source']],
            'Destination': [data['Destination']]
        })

        # We apply the same one-hot encoding used in training
        airline_dummies = pd.get_dummies(input_df[['Airline']], drop_first=True)
        source_dummies = pd.get_dummies(input_df[['Source']], drop_first=True)
        dest_dummies = pd.get_dummies(input_df[['Destination']], drop_first=True)

        # Concat
        input_df = pd.concat([input_df, airline_dummies, source_dummies, dest_dummies], axis=1)

        # Drop the original
        input_df.drop(columns=['Airline', 'Source', 'Destination'], inplace=True)

        # Align with training columns
        _, df_model = load_training_data()
        X = df_model.drop('Price', axis=1)
        all_cols = X.columns.tolist()

        # Ensure the input_df has all those columns (fill missing with 0)
        for col in all_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only the columns that the model expects
        input_df = input_df[all_cols]

        return input_df.values

    except Exception as e:
        st.error(f"Error in preprocessing input: {str(e)}")
        return None


# --------------------------------------------------------
# 6. Main Streamlit App
# --------------------------------------------------------
def main():
    st.title("✈️ Smart Flight Price Predictor")

    # Load data for analysis & model
    try:
        df_analysis, df_model = load_training_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Train or load model
    try:
        # Baseline model from scratch (or load saved model)
        baseline_model, X_train, X_test, y_train, y_test = train_baseline_model(df_model)
    except Exception as e:
        st.error(f"Error training/loading model: {str(e)}")
        return

    # Tabs
    tabs = st.tabs(["Price Prediction", "Data Analysis", "Model Evaluation", "Airlines News"])

    # ------------------------------
    # Tab 1: Price Prediction
    # ------------------------------
    with tabs[0]:
        st.subheader("Enter Flight Details")

        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date of Journey")
            Date_of_Journey = date.strftime("%d/%m/%Y")

            Airline = st.selectbox("Airline", [
                "Air India", "GoAir", "IndiGo", "Jet Airways", "Jet Airways Business",
                "Multiple carriers", "Multiple carriers Premium economy", "SpiceJet",
                "Trujet", "Vistara", "Vistara Premium economy"
            ])

            Source = st.selectbox("Source", ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
            Destination = st.selectbox("Destination", ["Banglore", "Cochin", "Delhi",
                                                       "Hyderabad", "Kolkata", "New Delhi"])

        with col2:
            Total_Stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stop", "3 stop", "4 stop"])
            dep_time = st.time_input("Departure Time")
            arr_time = st.time_input("Arrival Time")
            Duration = st.text_input("Duration (e.g., 2h 50m)", "2h 50m")

        # Build input data
        input_data = {
            'Date_of_Journey': Date_of_Journey,
            'Airline': Airline,
            'Source': Source,
            'Destination': Destination,
            'Total_Stops': Total_Stops,
            'Dep_Time': dep_time.strftime("%H:%M"),
            'Arrival_Time': arr_time.strftime("%H:%M"),
            'Duration': Duration
        }

        # Predict button
        if st.button("Predict Price", key="predict"):
            with st.spinner("Calculating..."):
                processed_data = preprocess_input_for_inference(input_data)
                if processed_data is not None:
                    price_pred = baseline_model.predict(processed_data)
                    st.success(f"💰 Predicted Price: ₹{price_pred[0]:,.2f}")

    # ------------------------------
    # Tab 2: Data Analysis
    # ------------------------------
    with tabs[1]:
        st.header("Exploratory Data Analysis")

        # Dark-themed Plotly boxplots
        st.subheader("Dark Themed Boxplots")
        colA, colB, colC = st.columns(3)
        with colA:
            fig_airline_dark = create_dark_boxplot_by_airline(df_analysis)
            if fig_airline_dark:
                st.plotly_chart(fig_airline_dark, use_container_width=True)
            else:
                st.write("No 'Airline' column found for dark boxplot.")

        with colB:
            fig_source_dark = create_dark_boxplot_by_source(df_analysis)
            if fig_source_dark:
                st.plotly_chart(fig_source_dark, use_container_width=True)
            else:
                st.write("No 'Source' column found for dark boxplot.")

        with colC:
            fig_stops_dark = create_dark_boxplot_by_stops(df_analysis)
            if fig_stops_dark:
                st.plotly_chart(fig_stops_dark, use_container_width=True)
            else:
                st.write("No 'Total_Stops' column found for dark boxplot.")

        # Additional Seaborn/Matplotlib plots
        st.subheader("Classic Seaborn/Matplotlib Plots")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price Distribution (Histogram)")
            fig_price_dist = create_price_distribution_plot(df_analysis)
            st.pyplot(fig_price_dist)

            st.subheader("Stops Analysis (Boxplot)")
            fig_stops = create_stops_analysis_plot(df_analysis)
            st.pyplot(fig_stops)

        with col2:
            st.subheader("Airline Pricing (Boxplot)")
            fig_airline = create_airline_price_plot(df_analysis)
            if fig_airline:
                st.pyplot(fig_airline)

            st.subheader("Feature Importance (ExtraTrees)")
            fig_feat_import = create_feature_importance_plot(df_model)
            if fig_feat_import:
                st.pyplot(fig_feat_import)

        st.subheader("Correlation Heatmap")
        fig_corr = create_correlation_heatmap(df_model)
        st.pyplot(fig_corr)

    # ------------------------------
    # Tab 3: Model Evaluation
    # ------------------------------
    with tabs[2]:
        show_model_evaluation(baseline_model, X_test, y_test, title="Baseline RandomForest Model Performance")

        st.markdown("---")
        st.markdown("### Train a Tuned Model (Optional)")
        st.markdown(
            "Below is a demonstration of how you'd tune and evaluate a Random Forest model. "
            "This can take extra time if the dataset is large."
        )
        if st.button("Run RandomizedSearchCV"):
            with st.spinner("Tuning hyperparameters..."):
                n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
                max_features = ['auto', 'sqrt']
                max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
                min_samples_split = [2, 5, 10, 15, 100]
                min_samples_leaf = [1, 2, 5, 10]

                random_grid = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }

                random_rf = RandomizedSearchCV(
                    estimator=baseline_model,
                    param_distributions=random_grid,
                    scoring='neg_mean_squared_error',
                    n_iter=10,
                    cv=5,
                    verbose=2,
                    random_state=42,
                    n_jobs=-1
                )

                random_rf.fit(X_train, y_train)
                best_model = random_rf.best_estimator_

                st.write("**Best Params:**", random_rf.best_params_)

                show_model_evaluation(best_model, X_test, y_test, title="Tuned RandomForest Model Performance")

    # ------------------------------
    # Tab 4: Airlines News
    # ------------------------------
    with tabs[3]:
        st.header("Latest Airline News")
        news_items = fetch_airline_news()
        if isinstance(news_items, str):
            st.error(news_items)
        elif news_items:
            for item in news_items:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.write(f"*Published on: {item['published']}*")
                st.write("---")
        else:
            st.write("No news available at the moment.")


# --------------------------------------------------------
# Run the app
# --------------------------------------------------------
if __name__ == "__main__":
    main()
