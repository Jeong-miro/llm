# 제품명, 브랜드명, 브랜드핵심가치, 제품특징, 필수포함키워드 => 1~2줄의 문구 5개 정도 추천
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr

load_dotenv(find_dotenv())
MODEL = "gpt-6-luna"

client = OpenAI()


def ad_generator(name,brand_name,value,feature,keyword):

    # if 문으로 미입력 확인 가능

    context = [
    {"role":"user","content":""},
    {"role":"user","content":""},
    {"role":"user","content":""},
    {"role":"user","content":""},
    {"role":"user","content":""},
        
    ]

    instructions="""
    당신은 최고의 카피라이터입니다..
    사용자가 제시한 문구를 활용하여 광고문구를 작성해주세요.
"""

    response = client.responses.create(
    model=MODEL,
    instructions=instructions,
    input=f"""
    아래 내용을 참고해서 1~2줄짜리 광고 문구 5개를 생성해주세요.
    - 제품명 : {name}
    - 브랜드명 : {brand_name}
    - 브랜드핵심가치 : {value}
    - 제품특징 : {feature}
    - 필수포함키워드 : {keyword}
""",
)
    
    return response.output_text

demo = gr.Interface(
    fn=ad_generator,
    inputs=[
        gr.Text(label="제품명"),
        gr.Text(label="브랜드명"),
        gr.Text(label="브랜드핵심가치"),
        gr.Text(label="제품특징"),
        gr.Text(label="필수포함키워드"),
            ],
    outputs=gr.Markdown(),
    title="📜 OpenAI 광고 문구 추천 프로그램",
    description="OpenAI-6-luna 기반 광고 문구 추천 서비스",
)

demo.launch()