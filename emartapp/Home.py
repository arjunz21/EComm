import os
import warnings
import pandas as pd
import streamlit as st
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

datapath = os.getcwd() + "\\Notebooks\\data\\EMart.csv"
df = pd.read_csv(datapath)


st.set_page_config(page_title="EMart AiML App", page_icon="", layout="wide", initial_sidebar_state="expanded")
st.title("EMart Ai-ML App")

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
orders.metric("# Orders", len(df['OrderNumber']), "")
qty.metric("# Quantity", df['Quantity'].sum(), "")
sales.metric("Sales", df['Sales'].sum(), "-8%")
profit.metric("Profit", df['Profit'].sum(), "-8%")


date1, date2 = st.columns(2)
date1.date_input('Start Date')
date2.date_input('End Date')
st.button("Start Analysis")

# Main App
st.header('Dashboard')

stats, dav, profile = st.tabs(['Statistical Data', 'Data Visualization', 'Data Profiling'])

with stats:
    st.subheader("")
    st.write(df.head())
    st.write(df.describe().T)
    st.write(df.describe(include='object').T.sort_values(by='unique'))
    missing_value = pd.DataFrame({
        'TotalMissing' : df.isnull().sum(),
        'Percentage' : (df.isnull().sum()/len(df))*100 })
    st.write(missing_value[missing_value['Percentage'] > 0].sort_values(by='Percentage',ascending=False))


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
        profile = ProfileReport(df)
        st.success("File uploaded successful")

        st.title("Detailed Report of uploaded data")
        st.write(df.head())
        st_profile_report(profile)
    else:
        st.write("File is not uploaded")


# st.balloons()