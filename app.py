import os
import requests
import warnings
import pandas as pd
import streamlit as st
#import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
#from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

datapath = os.path.dirname(os.getcwd()) + "\\ECOMM\\Notebooks\\data\\EMart.csv"
df = pd.read_csv(datapath)
df.drop(columns=['Unnamed: 0'], inplace=True)


st.set_page_config(page_title="EMart AiML App", page_icon="", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: yellow;'>EMart Ai-ML App</h1>", unsafe_allow_html=True)

with st.expander("How to Use the app"):
    st.warning("Please select other fields in sidebar then select dates for proper working")

with st.sidebar:
    continent = st.multiselect('Continent', default=df["Continent"].unique(), options=df["Continent"].unique())
    country = st.multiselect('Country', default=df["Country"].unique(), options=df["Country"].unique())
    state = st.multiselect('State', default=[], options=df["State"].unique())
    city = st.multiselect('City', default=[], options=df["City"].unique())
    currency = st.multiselect('Currency', default=df["Currency"].unique(), options=df["Currency"].unique())
    brand = st.multiselect('Brand', default=[], options=df["Brand"].unique())

orders, qty, sales, profit = st.columns(4)
orders.metric("# Orders", len(df['OrderDate']), "")
qty.metric("# Quantity", df['Quantity'].sum(), "")
sales.metric("Sales", df['Sales'].sum(), "8%")
profit.metric("Profit", df['Profit'].sum(), "8%")


date1, date2 = st.columns(2)
date1.date_input('Start Date')
date2.date_input('End Date')
st.button("Start Analysis")

# Main App
st.markdown("<h1 style='text-align: center; color: yellow;'>Dashboard</h1>", unsafe_allow_html=True)

salespred, stats, dav, profile = st.tabs(['Sales Prediction', 'Statistical Data', 'Data Visualization', 'Data Profiling'])

with salespred:
    st.markdown("<h3 style='text-align: center; color: white;'>Sales Prediction using Custom ML-API</h3>", unsafe_allow_html=True)
    tdata, getModel, testmodel = st.columns(3)

    with tdata:
        st.info("Get Data & Transform")
        with st.form(key="transformdata"):
            inpdata = {}
            inpdata['trainpath'] = st.text_input(label="Train CSV PAth")
            inpdata['valpath'] = st.text_input(label="Validation CSV Path")
            inpdata['testpath'] = st.text_input(label="Test CSV Path")
            inpdata['ordcolumn'] = st.text_input(label="Ordinal Column Name")
            inpdata['targetcolumn'] = st.text_input(label="Target Column Name")
            
            getdatabutton = st.form_submit_button('GetData')
            submitButton = st.form_submit_button("TransformData")

            if submitButton:
                st.write(inpdata)
                res = requests.post("http://localhost:8000/api/emart/transformdata", json=inpdata)
                st.write("result:", str(res.status_code))
                st.write("data:", str(res.json()))
            elif getdatabutton:
                res = requests.get("http://localhost:8000/api/emart/getdata")
                if res.status_code == 200:
                    st.write("New Data Fetched")
    
    with getModel:
        st.info("Find the Best Model & Train")
        with st.form(key="getBestModel"):
            inpdata = {}
            inpdata['prepath'] = st.text_input(label="PreProcessor Path")
            inpdata['Xtrpath'] = st.text_input(label="XTrain Path")
            inpdata['ytrpath'] = st.text_input(label="YTrain Path")
            inpdata['Xvalpath'] = st.text_input(label="XValidation Path")
            inpdata['yvalpath'] = st.text_input(label="YValidation Path")

            submitButton = st.form_submit_button("GetBestModel")
            trainButton = st.form_submit_button("TrainModel")

            if submitButton:
                st.write(inpdata)
                res = requests.post("http://localhost:8000/api/emart/getBestModel", json=inpdata)
                st.write("data:", str(res.json()))
            elif trainButton:
                res = requests.post("http://localhost:8000/api/emart/train", json=inpdata)
                st.write("result:", str(res.status_code))
    
    with testmodel:
        st.info("Test the Model & Get Metrics")
        if st.button('GetAllModelScores'):
            res = requests.get("http://localhost:8000/api/emart/getallscores")
            st.write("result:", str(res.status_code))
            st.write("data:", str(res.json()))
        
        if st.button('GetModelScores'):
            res = requests.get("http://localhost:8000/api/emart/scores")
            st.write("result:", str(res.status_code))
            st.write("data:", str(res.json()))

    
    st.markdown("<h3 style='text-align: center; color: white;'>Prediction Model using ML-API</h3>", unsafe_allow_html=True)
    with st.form(key="sales_form"):
        col1, col2, col3 = st.columns(3)
        inpdata = {}
        
        with col1:
            inpdata['Name'] = st.text_input(label="Customer Name")
            inpdata['Age'] = st.selectbox("Age", options=df["Age"].unique(), index=None)
            inpdata['Quantity'] = st.number_input(label="Quantity")
            inpdata['OrderDay'] = str(st.slider("Order Day", 0, 30, 1))
            inpdata['OrderMonth'] = str(st.slider("Order Month", 0, 12, 1))
        
        with col2:
            inpdata['ProductName'] = st.text_input(label="Product Name")
            inpdata['Brand'] = st.selectbox("Brand", options=df["Brand"].unique(), index=None)
            inpdata['Color'] = st.text_input(label="Color")
            inpdata['Category'] = st.text_input(label="Category")
            inpdata['Subcategory'] = st.text_input(label="Sub Category")
        
        with col3:
            inpdata['Currency'] = st.selectbox("Currency", options=df["Currency"].unique(), index=None)
            inpdata['Continent'] = st.selectbox("Continent", options=df["Continent"].unique(), index=None)
            inpdata['State'] = st.selectbox("State", options=df["State"].unique(), index=None)
            inpdata['Country'] = st.selectbox("Country", options=df["Country"].unique(), index=None)
            inpdata['City'] = st.selectbox("City", options=df["City"].unique(), index=None)
            

        submitButton = st.form_submit_button("Predict")
        if submitButton:
            st.write(inpdata)
            res = requests.post("http://localhost:8000/api/emart/predict?preprocessPath=artifacts%5Cpreprocessor.pkl&modelPath=artifacts%5Cmodel.pkl",
                                json=inpdata)
            st.write("result:", str(res.status_code))
            st.write("data:", str(res.json()))



with stats:
    st.subheader("Sample Data")
    st.write(df.head())
    col1, col2, col3 = st.columns(3)

    col1.write(df.describe().T)
    col2.write(df.describe(include='object').T.sort_values(by='unique'))
    missing_value = pd.DataFrame({
        'TotalMissing' : df.isnull().sum(),
        'Percentage' : (df.isnull().sum()/len(df))*100 })
    col3.write(missing_value[missing_value['Percentage'] > 0].sort_values(by='Percentage',ascending=False))


with dav:
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], format="%m/%d/%Y", errors='coerce')

    # 1. Sales by Category
    sales_by_category = df.groupby('Category')['Quantity'].sum().reset_index()
    trace1 = go.Bar(x=sales_by_category['Category'], y=sales_by_category['Quantity'], name='Sales by Category')

    # Sales Trend Over Time
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').dt.to_timestamp()
    sales_trend = df.groupby('OrderMonth')['Quantity'].sum().reset_index()
    trace2 = go.Scatter(
        x=sales_trend['OrderMonth'],
        y=sales_trend['Quantity'],
        mode='lines+markers',
        name='Sales Trend Over Time'
    )

    # Customer Demographics - Age Distribution
    trace3 = go.Histogram(x=df['Age'], name='Customer Age Distribution')

    # Customer Demographics - Gender Distribution
    trace4 = go.Pie(labels=df['Gender'].value_counts().index,
                    values=df['Gender'].value_counts().values, name='Customer Gender Distribution')

    # Sales Distribution
    trace5 = go.Histogram(x=df['Sales'], name='Sales Distribution')

    # Profit Distribution
    trace6 = go.Histogram(x=df['Profit'], name='Profit Distribution')

    # Quantity Distribution
    trace7 = go.Histogram(x=df['Quantity'], name='Quantity Distribution')

    # Currency Distribution
    trace8 = go.Histogram(x=df['Currency'], hoverinfo='x+text', name='Currency Distribution')

    # Country Distribution
    trace9 = go.Histogram(x=df['Country'], hoverinfo='x+text', name='Country Distribution')

    # Continent Distribution
    trace10 = go.Histogram(x=df['Continent'], hoverinfo='x+text', name='Continent Distribution')

    # Subcategory Distribution
    trace11 = go.Histogram(x=df['Subcategory'], hoverinfo='x+text', name='Subcategory Distribution')

    # Correlation Matrix
    corr_matrix = df[['Quantity', 'Sales', 'Profit']].corr()
    trace12 = go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                        colorscale='Viridis', showscale=False, zmin=-1, zmax=1)

    # Create subplots
    fig = make_subplots(rows=4, cols=3, specs=[[{}, {}, {}], [{"type": "pie"}, {}, {}], [{}, {}, {}], [{}, {}, {}]],
                        subplot_titles=(
                            'Sales by Category', 'Sales Trend Over Time', 'Customer Age Distribution', 'Customer Gender Distribution',
                            'Sales Distribution', 'Profit Distribution', 'Quantity Distribution', 'Currency Distribution',
                            'Country Distribution', 'Continent Distribution', 'Subcategory Distribution', 'Correlation Matrix'))

    # Add traces to subplots
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=1, col=3)
    fig.add_trace(trace4, row=2, col=1)
    fig.add_trace(trace5, row=2, col=2)
    fig.add_trace(trace6, row=2, col=3)
    fig.add_trace(trace7, row=3, col=1)
    fig.add_trace(trace8, row=3, col=2)
    fig.add_trace(trace9, row=3, col=3)
    fig.add_trace(trace10, row=4, col=1)
    fig.add_trace(trace11, row=4, col=2)
    fig.add_trace(trace12, row=4, col=3)

    # Update layout for the overall figure
    fig.update_layout(height=1200, width=1800, title_text="Dashboard", showlegend=True)

    # Display the figure in Streamlit
    st.plotly_chart(fig)


with profile:
    uploaded_file = st.file_uploader("Upload your file here")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        #profile = ProfileReport(df)
        st.success("File uploaded successful")

        st.title("Detailed Report of uploaded data")
        st.write(df.head())
        #st_profile_report(profile)
    else:
        st.write("File is not uploaded")


# st.balloons()