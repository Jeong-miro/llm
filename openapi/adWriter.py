# 제품명,브랜드명,브랜드핵심가치,제품특징,필수포함키워드 => 광고문구 1~2줄 5개 정도 추천

from dotenv import load_dotenv, find_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv(find_dotenv())
MODEL = "gpt-6-luna"

client = OpenAI()


# 5개
def ad_generator(name, brand_name, value, features, keyword):
    
    instructions = """당신은 최고의 카피라이터입니다.사용자가 제시한 문구를 활용하여 광고문구를 작성하세요"""

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=f"""
    아래 내용을 참고해서 1~2줄짜리 광고 문구 5개 생성해줘.
    - 제품명 : {name}
    - 브랜드명 : {brand_name}
    - 브랜드 핵심 가치 : {value}
    - 제품 특징 : {features}
    - 필수 포함 키워드 : {keyword}
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
    title="📜 OpenAI 광고 문구 프로그램",
    description="OpenAI-luna 기반 광고 문구 서비스",
)

demo.launch(debug=True)
