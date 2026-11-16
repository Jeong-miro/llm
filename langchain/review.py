from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

import gradio as gr

load_dotenv(find_dotenv())
gpt_model = init_chat_model(model="gpt-6-luna")

# ReviewAnalysis
# sentiment
# score
# pros : 긍정적인 점 목록
# cons : 부정적인 점 목록
# recommend : 추천 여부 True ? False
# reply : 판매자  입장의 고객 답변 문장

class ReviewAnalysis(BaseModel):
    sentiment:Literal['positive','negative','neutral']
    score:float=Field(ge=0.0, le=1.0, description="감정 강도")
    pros:list[str]=Field(description="긍정적인 점 목록")
    cons:list[str]=Field(description="부정적인 점 목록")
    recommend:bool = Field(description="추천여부 True/False")
    reply:str=Field(description="판매자 입장의 고객 답변 문장")

pydantic_parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt = ChatPromptTemplate.from_messages(
[
    ("system","당신은 상품 리뷰 분석 전문가 입니다{format_instructions}"),
    ("human","다음 리뷰를 분석하세요 {text}"),
]).partial(format_instructions=pydantic_parser.get_format_instructions())

chain = prompt | gpt_model | pydantic_parser

def analyze(text):
    reviews = [line.strip() for line in text.split("===") if line.strip()]

    if not reviews:
        return "리뷰를 입력하세요"
    
    response = chain.batch([{"text":r} for r in reviews])

    output = []
    for i ,(text, r) in enumerate(zip(reviews,response),1):

        emoji = {'positive':'😊','negative':'☹️','neutral':'😐'}[r.sentiment]
        output.append (f'[리뷰 {i}]')
        output.append (f'[원 리뷰 {text}]')
        output.append (f'{emoji} [{r.sentiment}] {r.score:.2f}')
        output.append (f'장점 {r.pros}')
        output.append (f'단점 {r.cons}')
        output.append (f'추천여부 {'추천' if r.recommend else '비추천'}')
        output.append (f"판매자 답변 {r.reply}")

    return "\n".join(output)

# 출력문
# [리뷰 2]
# 감정 ☹️😊😐
# 리뷰 원문
# 장점
# 단점
# 추천여부
# 판매자 답변

demo = gr.Interface(fn=analyze, inputs=gr.Textbox(
    label="리뷰 입력 (여러 개는 === 으로 구분)",
    lines=10,
    placeholder="배송이 빠르고 포장도 꼼꼼했어요"
), outputs=gr.Textbox(
    label="분석 결과",
    lines=20,
) ,title="상품 리뷰 분석")
demo.launch()