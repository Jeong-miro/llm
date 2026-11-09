# 텍스트 => 요약, 키워드 추출, 분류, 감정분석, 어조 분석

from ollama import Client
import gradio as gr

OLLAMA_HOST="http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)


def analyze(image, question):

    # 이미지가 없다면 "이미지를 업로드 해 주세요" return
    if image is None:
        return "이미지를 업로드 해 주세요"

    # 모델에게 요청
    messages = [
        {"role":"system","content":"당신은 전문 이미지 분석 ai 입니다."},
    {"role":"user","content":f"{question}",  "images":[image]},
    ]


    response = client.chat(model=MODEL,messages=messages,stream=False,think=False)
    
    return response.message.content

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="filepath", label="이미지업로드"),
            gr.Textbox(value="이미지에 무엇이 있는지 자세히 설명해줘", label="질문"),
            ],
    outputs=gr.Markdown(),
    title="AI Image Analyzer",
    description="Qwen3.5:9B 기반 이미지 분석 서비스",
)

demo.launch()