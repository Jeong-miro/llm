import gradio as gr
from transformers import pipeline
import torch

clf_ko = pipeline(
    "text-classification",
    model="tabularisai/multilingual-sentiment-analysis",
    device_map="auto",
    dtype=torch.float16,
    top_k=2,
)


def predict_sentiment(text):
    # [[]]
    result = clf_ko(text)[0]
    # [{'label': 'Positive', 'score': 0.3806661367416382}, {'label': 'Very Positive', 'score': 0.24776466190814972}]
    print(result)
    predict = {}

    for item in result:
        label = item["label"]
        predict[label] = item["score"]

    return predict


app = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="문장을 입력하세요"),
    outputs=gr.Label(),
    title="AI 감정분석",
    api_name="predict",
)

app.launch(debug=True)
