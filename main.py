# pip install streamlit yfinance plotly tensorflow
import streamlit as st
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

# import library pandas
import pandas as pd

# Import library numpy
import numpy as np

# Import library matplotlib untuk visualisasi
import matplotlib.pyplot as plt

# import library for build model 
from tensorflow import keras


# import library untuk data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

START = "2013-11-19"
END = "2021-11-19"

st.title('Stock Forecast App')

stocks = ('BBCA.JK', 'BBNI.JK', 'BBRI.JK', 'BDMN.JK', 'BMRI.JK', 'BNGA.JK', 'PNBN.JK')
selected_stock = st.selectbox('Pilih dataset ', stocks)


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

model_toload = ('RNN', 'LSTM')
selected_model= st.selectbox('Pilih model ', model_toload)

def load_model(stock_name,model_name):
	filepath = "models/"+stock_name+"_"+model_name+".h5"
	model = keras.models.load_model(filepath)

	return model
	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')
model_load_state = st.text('Loading model...')
model_set = load_model(selected_stock,selected_model)
model_load_state.text('Loading model... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	#fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Stock '+selected_stock, xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# PEMODELAN SAHAM 

data = data.dropna()

# Kolom 'close' yang akan kita gunakan dalam membangun model
# Slice kolom 'close' +
close_data = data['Close'].values

# Menskalakan data antara 1 dan 0 (scaling) pada close data
scaler = MinMaxScaler(feature_range=(0,1))
close_scaled = scaler.fit_transform(close_data.reshape(-1,1))

# definisikan variabel step dan train 
step_size = 21
train_x = []
train_y = []

# membuat fitur dan lists label
for i in range(step_size,len(close_scaled)-1):                
	train_x.append(close_scaled[i-step_size:i,0])
	train_y.append(close_scaled[i,0])

# mengonversi list yang telah dibuat sebelumnya ke array
train_x = np.array(train_x)                  
train_y = np.array(train_y)                           

# 373 hari terakhir akan digunakan dalam pengujian
# 1600 hari pertama akan digunakan dalam pelatihan
test_x = train_x[1600:]            
train_x = train_x[:1600]           
test_y = train_y[1600:]  
train_y = train_y[:1600]

# reshape data untuk dimasukkan kedalam Keras model
train_x = np.reshape(train_x, (len(train_x), step_size, 1))       
test_x = np.reshape(test_x, (len(test_x), step_size, 1))                        

# Score model
predictions = model_set.predict(test_x)
score = r2_score(test_y,predictions)
st.write("R2 Score: ",score)
mse = mean_squared_error(test_y,predictions)
st.write("MSE Score: ",mse)

predictions = scaler.inverse_transform(predictions)
test_y = scaler.inverse_transform(test_y.reshape(-1,1))

# Visualisasi Metode RNN dan LSTM
data_index = list(range(0,len(predictions)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_index,y=np.array(test_y)[:,0], name="real"))
fig.add_trace(go.Scatter(x=data_index, y=np.array(predictions)[:,0], name="prediction"))
fig.layout.update(title_text=selected_model+' model', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)