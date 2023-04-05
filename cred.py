import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Set up the page title and layout
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon=":money_with_wings:")

# Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess the dataset
    rbs = RobustScaler()
    df_small = df[['Time', 'Amount']]
    df_small = pd.DataFrame(rbs.fit_transform(df_small))
    df_small.columns = ['scaled_time', 'scaled_amount']
    df = pd.concat([df, df_small], axis=1)
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Create a plot of class distribution
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Class'].value_counts().index, y=df['Class'].value_counts().values))
    fig.update_layout(title_text="Count of fraud and non-fraud transactions")
    st.plotly_chart(fig)

    # Create a plot of amount vs class
    st.subheader("Amount vs Class")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[df['Class']==0]['scaled_amount'], y=df[df['Class']==0]['Class'], mode='markers', name='Non-Fraud'))
    fig.add_trace(go.Scatter(x=df[df['Class']==1]['scaled_amount'], y=df[df['Class']==1]['Class'], mode='markers', name='Fraud'))
    fig.update_layout(title_text="Amount vs Class")
    st.plotly_chart(fig)

    # Create a balanced dataset
    non_fraud = df[df['Class']==0].sample(frac=1)[:492]
    fraud = df[df['Class']==1]
    new_df = pd.concat([non_fraud, fraud]).sample(frac=1)

    # Split the dataset into training and testing sets
    X = new_df.drop('Class', axis=1)
    y = new_df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # Train and evaluate a logistic regression model
    st.subheader("Logistic Regression")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    st.write("Classification Report")
    st.write(classification_report(y_test, pred))
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))
    st.write("Accuracy")
    st.write(round(accuracy_score(y_test, pred)*100, 2))

    # Train and evaluate a decision tree classifier
    st.subheader("Decision Tree Classifier")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    st.write("Classification Report")
    st.write(classification_report(y_test, pred))
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))
    st.write("Accuracy")
    st.write(round(accuracy_score(y_test, pred)*100, 2))

    # Train and evaluate a random forest classifier
    st.subheader("Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    st.write("Classification Report")
    st.write(classification_report(y_test, pred))
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))
    st.write("Accuracy")
    st.write(round(accuracy_score(y_test, pred)*100, 2))