# 글쓰기 도우미 서비스
# 격식체, 친근한 말투, 비즈니스 문체, 간결한 문체, 전문적인 문체 => 5개의 스타일별로 문장 완성
# 원본문장 : 오늘 아파서 회사를 못가겠다 => 격식체

from ollama import Client
import gradio as gr

OLLAMA_HOST="http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)


def rewrite_text(text, style):

    content = f"""
    너는 전문 한국어 글쓰기 도우미야.

    사용자가 작성한 문장을 다음 스타일로 변경해 줘.
    스타일 : {style}

    원문의 핵심 의미는 유지하고 자연스럽게 작성해.

"""
    
    # 요약

    messages = [
        {"role":"system","content":content},
    {"role":"user","content":f"{text}"},
    ]


    response = client.chat(model=MODEL,messages=messages,stream=False,think=False)
    
    return response.message.content

demo = gr.Interface(
    fn=rewrite_text,
    inputs=[
        gr.Textbox(lines=7,label="원본 문장"),
            gr.Radio(choices=['격식체','친근한 말투','비즈니스 문체','간결한 문체','전문적인 문체'],
                        value='격식체',
                        label='변경할 스타일',
                        ),
            ],
    outputs=gr.Textbox(lines=7,label="변환 결과 문장"),
    title="AI Write Assitant",
    description="Qwen3.5:9B 기반 AI 글쓰기 도우미 서비스",
)

demo.launch()