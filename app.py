import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
import mlem


def plot():
    st.title ("Grafici")

    option = st.radio (
    'Scegli il grafico da visualizzare',
    ('Correlation Matrix', 'Pair Plot',), horizontal=True)

    if option == 'Correlation Matrix':
        df = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv')
        df.rename(columns={"medv":"price"}, inplace=True)
        df.drop(df.tail(1).index,inplace=True)
        s = df.select_dtypes(include='object').columns
        df[s] = df[s].astype("float")
        fig,ax=plt.subplots()
        sns.heatmap(df.corr(),annot=True)
        st.write(fig)

    elif option == 'Pair Plot':
        df = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/formart_house.csv')
        df.rename(columns={"medv":"price"}, inplace=True)
        df.drop(df.tail(1).index,inplace=True)
        s = df.select_dtypes(include='object').columns
        df[s] = df[s].astype("float")
        pairplot=sns.pairplot(df, hue='price',height=3, aspect=1)
        st.pyplot(pairplot)



#def show_footer():
    #st.markdown("")


def main():
    st.title(":blue[PROVA SOMMATIVA DEL WALLONE NAZIONALE] 	:chart_with_upwards_trend:")
    st.title ("Inserier valori da predirre")


    crim = st.number_input('Inserisci indice crim',0.0,1000.0,3.613524)
    zn = st.number_input('Inserisci indice zn',0.0,1000.0,11.363636)
    indus = st.number_input('Inserisci indice indus',0.0,1000.0,11.136779)
    chas = st.number_input('Inserisci indice chas',0.0,1000.0,0.069170)
    nox = st.number_input('Inserisci indice nox',0.0,1000.0,0.554695)
    rm = st.number_input('Inserisci indice rm',0.0,1000.0,6.284634)
    age = st.number_input('Inserisci indice age',0.0,1000.0,68.574901)
    dis = st.number_input('Inserisci indice dis',0.0,1000.0,3.795043)
    rad = st.number_input('Inserisci indice rad',0.0,1000.0,9.549407)  
    tax = st.number_input('Inserisci indice tax',0.0,1000.0,408.237154)
    pratio = st.number_input('Inserisci indice pratio',0.0,1000.0,18.455534)    
    b = st.number_input('Inserisci indice b',0.0,1000.0,356.674032)
    lstat = st.number_input('Inserisci indice lstat',0.0,1000.0,12.653063) 


    new_model = mlem.api.load('model_.mlem')


    pred= new_model.predict([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,pratio,b,lstat]])
    st.write(round(pred[0],2))
    plot()

    
        
if __name__ == "__main__":
    main()
