import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import pickle

stock_name = 'HDFC'
#Building the Model
data = pd.read_csv("D:\\College\\ML\\project\\data\\HDFC.csv")
df = pd.DataFrame(data)

def get_corelated_col(cor_dat, threshold):
  feature = []
  value = []

  for i ,index in enumerate(cor_dat.index):
    if abs(cor_dat[index]) > threshold:
      feature.append(index)
      value.append(cor_dat[index])

  df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
  return df

corrdata = df.copy(['Prev Close','Open','High','Low','Last','Close','VWAP','Volume','Turnover','Deliverable Volume','%Deliverble'])
corrdata = corrdata.drop(['Series','Date','Symbol'],axis=1)
corrmap = corrdata.corr()

top_corelated_values = get_corelated_col(corrmap['Close'], 0.8)
df = df[top_corelated_values.index]
df.dropna(inplace=True)
df = df.iloc[-2500:]


#MODEL
X = df.drop(['Close'], axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

#dump a ML Model
with open ('classifier_hdfc.pkl','wb') as file:
  pickle.dump(model, file)

