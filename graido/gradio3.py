import gradio as gr
from transformers import pipeline
import torch
clf_ko = pipeline('text-classification',
             model='tabularisai/multilingual-sentiment-analysis',
             device_map="auto",dtype=torch.float16,
             top_k=2)


def predict_sentiment(text):
    # [{0 : 'Very Negative' ~ 2 : 'Neutral' ~ 4 : 'Very Positive'}]
    # 결과가 2차원 리스트로 나와서 한개만
    # [{'label': 'Negative', 'score': 0.5405002236366272}, 
    # {'label': 'Very Negative', 'score': 0.3422234058380127}]
    result = clf_ko(text)[0]
    print(result)
    predict = {}

    for item in result:
        label = item['label']
        predict[label] = item['score']
        #score = item['score']

    return predict

app = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3,placeholder="문장을 입력하세요"),
    outputs=gr.Label(),
    title="AI 감정분석",
    api_name="predict",
)

app.launch(debug=True)