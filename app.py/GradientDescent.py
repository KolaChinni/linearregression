import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
#-------------------------------Gradient Descent for single Feature------------------------------
def compute_cost(x,y,w,b):
  m=x.shape[0]
  sme=0
  for i in range(m):
    f_wb=w*x[i]+b
    j_wbcost=(f_wb-y[i])**2
    sme+=j_wbcost
  total_cost=(1/(2*m))*sme
  return total_cost

def compute_gradient(x,y,w,b):
  m=x.shape[0]
  dj_w=0
  dj_b=0
  for i in range(m):
    f_wb=w*x[i]+b
    dj_dw=(f_wb-y[i])*x[i]
    dj_db=f_wb-y[i]
    dj_w+=dj_dw
    dj_b+=dj_db
  dj_w=dj_w/m
  dj_b=dj_b/m
  return dj_w,dj_b

def gradient_descent(x,y,w,b,alpha,iter):
#  cost_hist=[]
#  wb_hist=[]
  for i in range(iter):
    dj_w,dj_b=compute_gradient(x,y,w,b)
    w=w-alpha*dj_w
    b=b-alpha*dj_b
#    if i<10000:
#      cost=compute_cost(x,y,w,b)
#      cost_hist.append(cost)
#      wb_hist.append([w,b])
  return w,b#,cost_hist,wb_hist


w=0
b=0
alpha=0.01
iter=10000

#taking only 500 examples from 1000000 data
#data=pd.read_csv(r'study_hour(singleLinearRegression)\synthetic_data.csv')
#data50000=data.sample(n=500)
#data50000.to_csv('sample_data.csv',index=False)
data=pd.read_csv('sample_data.csv')
dfdata=pd.DataFrame(data)
x=dfdata['Study Hours']
y=dfdata['Scores']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
w,b=gradient_descent(x_train,y_train,w,b,alpha,iter)
#print(cost_hist)
#print(wb_hist)
#predict=2
#print(w*predict+b)
def prediction(predict):
  return w*predict+b
st.title("Single Feature Prediction using GradientDescent")
st.subheader("Enter Data:")
hours=st.number_input("Hours",min_value=0.0,max_value=24.0)
if st.button("Predict Score"):
  predict=hours
  st.success(prediction(predict))

