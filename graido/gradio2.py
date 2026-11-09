import gradio as gr
from PIL import Image
from transformers import pipeline
import torch

vision = pipeline("image-text-to-text",
                   model="HuggingFaceTB/SmolVLM-256M-Instruct",
                   device_map="auto",
                   dtype=torch.float16)


def chat(message, history):
    #print("message : " ,message)

    # 메세지에서 텍스트와 이미지 분리
    text = message.get('text','').strip()
    files = message.get('files',[])

    # 1. 이미지가 없는 경우
    
    if not files:
        return '이미지가 없습니다.'
    # 2. 이미지가 있는 경우 : gradio 파일 경로 추출
    if files:
        image_file = files[0]

        # gradio 경로 추출
        # gradio 버전에 따라 파일이 dict 으로 전달되는 경우 처리
        # python type => int, str, bool, list, dict, tuple 
        if isinstance(image_file,dict):
            image_file = image_file.get("path")

        image = Image.open(image_file).convert("RGB")

        # 이미지 설명 요청(text)
        if not text:
            text = "Describe this image."

            
        messages = [
            {"role": "user", 
             "content": [
                {'type':'image','url':image},
                {'type':'text','text':text}
                ]}
            ]
        # 이미지에 대한 설명 생성 반환
        result = vision(text=messages, max_new_tokens=40,return_full_text=False)[0]['generated_text']

    
    return result.strip()


    #print("history : " ,history)
    

chatbot = gr.ChatInterface(fn=chat,multimodal=True,title="Chat Bot",description="이미지에 대한 text 를 등록하면 응답해주는 챗봇",
                        )
chatbot.launch(debug=True)

