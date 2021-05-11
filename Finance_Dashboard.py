import numpy as np
import pandas as pd 
import streamlit as st
import altair as alt
import investpy
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

html_temp = """
    <div style="background-color:ghostwhite;padding:10px;margin-bottom: 25px; border-radius: 35px; ">
    <h2 style="color:black;text-align:center; font-family: monospace">Moroccan Finance Dashboard</h2>
    <p class="child" style="position: absolute; bottom: 50px; right: 20px; webkit-transform: rotate(40deg);
    moz-transform: rotate(40deg); background-color:lightblue; padding-left: 10px; padding-right: 10px; border-radius: 5px">Updated Daily</p>
    <p style="color:black;text-align:center; font-family: monospace" > Is the Stock Market efficient ? </p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

Today = datetime.today().strftime('%d/%m/%Y')

df = investpy.indices.get_index_historical_data(index='Moroccan All Shares',
                                        country='Morocco',
                                        from_date='01/01/2015',
                                        to_date=Today)    


col1, col2 = st.beta_columns((1,2))

with col1:
    st.dataframe(df)
with col2:
    html_temp = """
    <div class="container">
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue; 
    background-color: #f0f2f6">  <b> What is The MASI index ?</b> 
    </div>
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue">The MASI index (Moroccan All Shares Index) is a stock index that tracks 
    the performance of all companies listed in the Casablanca Stock Exchange located at Casablanca, Morocco. It is one of 
    the two main indexes at the stock exchange, the other being the MADEX (Moroccan Most Active Shares Index)
    <a href="https://en.wikipedia.org/wiki/MASI_index" target="_blank" aria-pressed="true"> <i>Wikipedia</i> </a>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    html_temp = """
    <div class="container" style="margin-top: 20px">
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue; 
    background-color: #f0f2f6">  <b> What is Market Efficiency ? </b> 
    </div>
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue">Market efficiency refers to the
    degree to which market prices reflect all available, relevant information. If markets are efficient, then all 
    information is already incorporated into prices, and so there is no way to "beat" the market because there are no
    undervalued or overvalued securities available 
    <a href="https://www.investopedia.com/terms/m/marketefficiency.asp" target="_blank" aria-pressed="true"> <i>Investopedia</i> </a>
    <br>
    <b> In short</b>, if we can create a model that predicts the MASI index with high accuracy, the Moroccan Stock Market would
    be inefficient <i>(Weak form Inefficiency because we're using public information)</i>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


Date = df.index
df['Date']=Date

chart1 = alt.Chart(df).mark_line().encode(
    x=alt.X('Date',axis=alt.Axis(format='%Y-%m',labelAngle=-20)),
    y=alt.Y('Close',scale=alt.Scale(domain=[np.min(df.Close), np.max(df.Close)])), 
).properties(width=530, height=300, title="MASI")


Return = []

for i in range(1,len(df)):
    c = (df.Close[i]-df.Close[i-1])/df.Close[i-1]
    Return.append(c)

df = df.iloc[1:]
df['Return'] = Return

chart2 = alt.Chart(df).mark_line().encode(
    x=alt.X('Date',axis=alt.Axis(format='%Y-%m',labelAngle=-20)),
    y=alt.Y('Return',scale=alt.Scale(domain=[np.min(df.Return), np.max(df.Return)])), 
).properties(width=530, height=300, title='Return')

st.altair_chart(chart1 | chart2)

html_temp = """
    <div style="background-color:ghostwhite; padding:10px; padding-bottom: 4px; padding-top: 5px; margin-bottom: 25px">
    <p style="color:black;text-align:center; font-family: cursive; font-size=30px">Model Building</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


result_1 = adfuller(df.Close)

     
df['Difference'] = df['Close'].diff()
result_2 = adfuller(df.Difference[1:])
    

col1, col2 = st.beta_columns((1,1))

with col1:
    html_temp = """
    <div style="background-color:white;padding:10px;margin-bottom: 25px">
    <h5 style="color:black;text-align:center;">Stationarity Test - Augmented Dickey Fuller Test</h5>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    stationarity_df = pd.DataFrame({'Serie':['Close', 'First Difference'],
                                    'ADF Statistic':[result_1[0],result_2[0]],
                                    'p-value':[result_1[1],result_2[1]]})
    st.table(stationarity_df)
    
    st.markdown("<p style='text-align: center; font-family: monospace; font-size:12px'>The MASI index serie is stationary at first difference</p>", unsafe_allow_html=True)



with col2:
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date',axis=alt.Axis(format='%Y-%m',labelAngle=-20)),
        y=alt.Y('Difference',scale=alt.Scale(domain=[np.min(df.Difference), np.max(df.Difference)])), 
    ).properties(width=550, height=300)
    st.altair_chart(chart)  


MASI_diff_1 = df.Difference[1:]

model = ARIMA(df.Close, order=(3,1,2))
model_fit = model.fit()
results_summary = model_fit.summary()   


html_temp = """
    <div style="background-color:white">
    <p style="text-align:center; font-family:cursive">ARIMA(3,1,2)</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

col1, col2, col3 = st.beta_columns((1.125,1.16,2))

    
with col1:
    st.title('')
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_pacf(MASI_diff_1, lags=20, ax=ax)
    st.pyplot(fig)
    
with col2:
    st.title('')
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_acf(MASI_diff_1, lags=20, ax=ax)
    st.pyplot(fig)
    
with col3: 
    st.latex('W_t =  \phi_1 W_{t-1} + \phi_2 W_{t-2} + \phi_3 W_{t-3} + \\theta_{1} \epsilon_{t-1} + \\theta_2 \epsilon_{t-2}')
    

    results_as_html = results_summary.tables[1].as_html()
    summary = pd.read_html(results_as_html, header=0, index_col=0)[0]
    st.table(summary)

html_temp = """
    <div style="background-color:ghostwhite; padding:10px; padding-bottom: 4px; padding-top: 5px; margin-bottom: 25px">
    <p style="color:black;text-align:center; font-family: cursive; font-size=30px">Model validation</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)



col1, col2 = st.beta_columns((3,5))

with col1:
    pred = model_fit.get_prediction(start=pd.to_datetime('2020-01-02'), dynamic=False)
    pred_ci = pred.conf_int()
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax = df.Close['2015':].plot(label='observed', linewidth=2)
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, linewidth=2)
    
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='lightblue', alpha=.2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('MASI')
    plt.legend()
    
    st.pyplot(fig)
    
    results_as_html = results_summary.tables[2].as_html()
    summary = pd.read_html(results_as_html, header=0, index_col=0)[0]
    st.table(summary)
    
with col2:
    
    st.title('')
    pred = model_fit.get_prediction(start=pd.to_datetime('2020-01-02'), dynamic=False)
    pred_ci = pred.conf_int()
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax = df.Close['2020':].plot(label='observed', linewidth=4)
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, linewidth=4)
    
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='lightblue', alpha=.2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('MASI')
    plt.legend()
    
    st.pyplot(fig)
    


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    return mape

MAPE = round(forecast_accuracy(pred.predicted_mean, df.Close) * 100,2)
Accuracy = 100 - round(forecast_accuracy(pred.predicted_mean, df.Close) * 100,2) 

st.write('Around', MAPE,'% MAPE - Mean Absolute Percentage Error - implies the model is about', Accuracy, '% accurate in predicting the next observations.')


html_temp = """
    <div style="background-color:ghostwhite; padding:10px; padding-bottom: 4px; padding-top: 5px; margin-bottom: 25px">
    <p style="color:black;text-align:center; font-family: cursive; font-size=30px">Univariate Forecasting</p>
    </div>
    """

st.markdown(html_temp,unsafe_allow_html=True)



# Get forecast 500 steps ahead in future
pred_uc = model_fit.get_forecast(steps=100)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

col1, col2, col3= st.beta_columns((1,4,1))

with col2:
    Close = df.Close.tolist()
    figure, ax = plt.subplots()
    data = pd.DataFrame({'Close':Close})
    axes = data.Close.plot(label='observed', figsize=(20, 10))
    pred_uc.predicted_mean.plot(ax=axes, label='Forecast')
    axes.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='lightblue', alpha=.25)
    axes.set_xlabel('Date')
    axes.set_ylabel('MASI')
    
    plt.legend()
    st.pyplot(figure)
    
    

html_temp = """
    <div class="container" style="text-align: center">
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue; 
    background-color: #f0f2f6">  <b>Conclusion</b> 
    </div>
    <div id="story" name="story" rows="5" cols="33" style="font-size: .8rem; letter-spacing: 1px; padding: 10px; line-height: 1.5; border-radius: 5px; 
    border: 1px solid #ccc; box-shadow: 1px 1px 1px #999;border-left: 5px solid steelblue">We can accurately predict future values of the MASI index. As a consequence, The Moroccan Stock Market is not efficient,ie , Prices do not reflect all information.
    </div>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
