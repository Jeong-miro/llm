import gradio as gr
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 임베딩
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 모델
llm = init_chat_model(model="gpt-6-luna", reasoning={"effort": "low"})


def process_pdf(pdf_file):

    if pdf_file is None:
        return ("PDF 파일을 업로드해주세요", "", "", "", "")

    # 1. 문서로드
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 첫번째 페이지 내용
    first_page_content = docs[0].page_content

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 총 청크 수
    total_chunks = len(chunks)

    # 첫 번째 청크 내용
    first_chunk = chunks[0].page_content

    # 첫 번째 청크의 메타데이터
    first_chunk_metadata = chunks[0].metadata

    return (
        f"총 페이지 수 :  {len(docs)}",
        first_page_content,
        f"총 청크 수 {total_chunks}",
        first_chunk,
        str(first_chunk_metadata),
    )

def rag_chat(pdf_file, question):

    if pdf_file is None:
        return ("PDF 파일을 업로드해주세요", "")

    # 1. 문서 로드
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 2. 청킹 - 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # 총 청크 수
    total_chunks = len(chunks)

    # 4. 벡터스토어 저장 : FAISS, Chroma 
    vector_store = Chroma.from_documents(documents=chunks, 
                                        embedding=openai_embeddings, 
                                        persist_directory="./db/chroma_db",
                                        collection_name="research2")
    print('ChromaDB 저장 완료')
    # RAG(Retriever) : 벡터스토어에 내용을 검색 가능한 인터페이스로 변경
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retrieved_docs = retriever.invoke(question)

    # context 생성
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
    ('system','다음 컨텍스트를 참고하여 질문에 답하세요\n컨텍스트에 없는 내용은 모른다고 답하세요\n\n컨텍스트 : {context}'),
    ('human','{question}')
    ])

    # rag_chain = (
    # {
    # 'context': context,
    # 'question': RunnablePassthrough(),
    # } | prompt | llm | StrOutputParser())

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context":context,"question":question})
    # 근거, 답변
    retrieved_text = ""

    for i, doc in enumerate(retrieved_docs):
        retrieved_text += f"""
    [검색 문서 {i+1}]

    근거 내용:
    {doc.page_content}

    metadata:
    {doc.metadata}

    {'='*50}
"""

    return retrieved_text,answer



with gr.Blocks() as demo:
    gr.Markdown("# PDF RAG 테스트")

    with gr.Tabs():

        with gr.Tab("1단계 - PDF & Chunk"):
            pdf_input_1 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            run_btn_1 = gr.Button("분석시작")

            page_output = gr.Textbox(label="총 페이지 수")
            first_page_output = gr.Textbox(label="첫번째 페이지 내용", lines=10)
            chunk_count_output = gr.Textbox(label="총 Chunk 수")
            first_chunk_output = gr.Textbox(label="첫번째 Chunk 내용")
            metadata_output = gr.Textbox(label="첫번째 metadata 내용")

            run_btn_1.click(
                fn=process_pdf,
                inputs=pdf_input_1,
                outputs=[
                    page_output,
                    first_page_output,
                    chunk_count_output,
                    first_chunk_output,
                    metadata_output,
                ],
            )

        with gr.Tab("2단계 - PDF & QNA"):
            pdf_input_1 = gr.File(label="PDF 업로드", file_types=[".pdf"])
            
            question_input = gr.Textbox(label="질문 입력")
            run_btn_1 = gr.Button("질문하기")

            retrieved_output = gr.Textbox(label="검색된 chunk", lines=10)

            answer_output = gr.Textbox(label="최종 답변", lines=10)

            run_btn_1.click(
                fn=rag_chat,
                inputs=[pdf_input_1,question_input],
                outputs=[
                    retrieved_output,
                    answer_output,
                ],
            )

demo.launch()
