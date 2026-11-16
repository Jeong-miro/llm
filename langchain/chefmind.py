from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

load_dotenv(find_dotenv())
gpt_model = init_chat_model(model="gpt-6-luna")

def chat(message, history):
    system = """당신은 20년 경력의 전문 셰프이자 요리 연구가입니다.
    사용자의 요리 질문에 대해 재료, 조리방법, 실패  방지 팁
    대체 재료를 포함하여 답변하세요.
    항상 한국어로 답변하세요.
"""
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","{message}"),
    ])

    chain = prompt | gpt_model | StrOutputParser()
    response = chain.invoke({"message":message})
    return response

demo = gr.ChatInterface(fn=chat, title="요리 전문가 Bot")
demo.launch()