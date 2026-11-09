import gradio as gr
from transformers import pipeline
import torch

# 1. 모델 불러오기
clf_ko = pipeline('text-classification',
             model='tabularisai/multilingual-sentiment-analysis',
             device_map="auto",dtype="auto")

def predict_sentiment(text):
    # 형태 : [{'label' : 'Neagtive' , 'score' : 0.84385755436}] 
    result = clf_ko(text)

    label = result[0]['label']
    score = result[0]['score']
    '''
    text => 긍/부정 판단
    '''

    return f"감정 : {label}\n확률 : {score:.3f}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=["text"],
    outputs=["text"],
    api_name="predict"
)

demo.launch() 