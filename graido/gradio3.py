# 음성 비서 앱

# 음성 파일 업로드 (obama.mp3) => 텍스트 결과
# 텍스트 기반으로 질의 응답

import gradio as gr
from transformers import pipeline

# automatic-speech
whisper = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", return_timestamps=True
)
llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    dtype="auto",
    device_map="auto",
)

# 전역변수
voice_text = ""


def transcipt(path):
    global voice_text
    # path 넘겨서 텍스트로 변환 후 반환
    result = whisper(path)
    voice_text = result["text"]
    return voice_text


def question_answer(question):
    # 텍스트가 필요 => voice_text
    global voice_text

    if not voice_text:
        return "음성을 텍스트로 변환 후 질의하세요"

    msg = [
        {
            "role": "system",
            "content": f"다음 음성 내용을 참고해서 질문에 답변하세요. 음성내용: {voice_text}",
        },
        {"role": "user", "content": f"질문 : {question}"},
    ]
    return llm(msg, max_new_tokens=300, do_sample=False)[0]["generated_text"][-1][
        "content"
    ]


with gr.Blocks(title="AI Voice") as bot:
    gr.Markdown("## AI 음성 비서")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Audio(type="filepath")
            btn1 = gr.Button("텍스트 변환")
        with gr.Column(scale=2):
            output = gr.Textbox(label="음성출력", lines=3)

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Text(label="question")
            btn2 = gr.Button("질문하기")
        with gr.Column(scale=2):
            answer = gr.Text(label="answer", placeholder="답변이 표시됩니다.")

    btn1.click(fn=transcipt, inputs=inp, outputs=output, api_name="ai_voice")
    btn2.click(fn=question_answer, inputs=question, outputs=answer)

bot.launch(debug=True)
