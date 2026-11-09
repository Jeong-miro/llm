# 텍스트 => 요약, 키워드 추출, 분류, 감정분석, 어조 분석

from ollama import Client
import gradio as gr

OLLAMA_HOST="http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)


def analyze(text, task):

    prompts = {
               "요약": "다음 글을 핵심 위주로 요약해 줘",
               "키워드 추출": "다음 글의 핵심 키워드 5개를 추출해 줘",
               "분류": "다음 글을 주제와 문서 유형 분류해 줘",
               "감정 분석": "다음 글의 감정을 분석하고 근거 설명해 줘",
               "어조 분석": "다음 글의 어조와 문체를 분석해 줘",
               }
    
    # 요약

    messages = [
        {"role":"system","content":"당신은 전문 텍스트 분석 ai 입니다."},
    {"role":"user","content":f"{prompts[task]} 분석 텍스트:{text}"},
    ]


    response = client.chat(model=MODEL,messages=messages,stream=False,think=False)
    
    return response.message.content

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(lines=7,label="분석할 테스트"),
            gr.Dropdown(choices=['요약','키워드 추출','분류','감정 분석','어조 분석'],
                        value='요약',
                        label='분석작업',
                        ),
            ],
    outputs=gr.Markdown(),
    title="AI Text Analyzer",
    description="Qwen3.5:9B 기반 텍스트 분석 서비스",
)

demo.launch()