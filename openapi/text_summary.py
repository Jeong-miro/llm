# 텍스트 => 요약
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr

load_dotenv(find_dotenv())
OLLAMA_HOST="http://localhost:11434"
MODEL = "gpt-6-luna"

client = OpenAI()


def summarize_text(text):

    if text is None:
        return "텍스트를 입력해 주세요"

    instructions="""
    당신은 텍스트를 한국어로 요약하는 전문가 입니다.
    당신의 임무는 아래 주어진 텍스트 문장을 한국어로 요약하는 것 입니다.
    - 요약 시 아래 사항을 반드시 반영해야 합니다.
    - 중복된 내용은 생략하되, 반복되는 내용은 요약에서 더 강조합니다.
    - 사례 중심보다는 개념과 주장 중심으로 요약합니다.
    - 3줄 이내로 요약합니다.
    - 불릿 기호 형식으로 작성합니다.
"""

    response = client.responses.create(
    model=MODEL,
    instructions=instructions,
    input=text,
)
    
    return response.output_text

demo = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=7,label="요약할 텍스트"),
            ],
    outputs=gr.Markdown(),
    title="📜 OpenAI 요약 프로그램",
    description="OpenAI-6-luna 기반 텍스트 요약 서비스",
)

demo.launch()