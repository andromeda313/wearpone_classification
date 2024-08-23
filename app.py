import streamlit as st
import fastai
import pathlib
import plotly.express as px
import platform

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

plt = platform.system()
if plt =='Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title("Qurollarni classiffikasiya qilish")

file = st.file_uploader("Rasmni yuklash", 
                   type = ['png',
                           'jpeg', 
                           'jpg', 
                           'gif', 
                           'svg'])

if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('wearpons_model.pkl')
    
    prediction, pred_id, prob = model.predict(img)
    st.success(f"Bashorat:{prediction}")
    st.info(f'Ehtimollik: {prob[pred_id]*100:.1f} %')
    
    fig = px.bar(x = prob*100, y = model.dls.vocab)
    st.plotly_chart(fig)
