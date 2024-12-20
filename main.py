import streamlit as st
import os
import base64
from typing import List, Tuple
from PyPDF2 import PdfReader
from PIL import Image
import io
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from dotenv import load_dotenv
import warnings
import concurrent.futures
import shutil
import hashlib
import time
import pytesseract
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv('.env', override=True)

# Azure OpenAI 설정
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure OpenAI Embedding 설정
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

# Azure Computer Vision 설정 추가
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF 챗봇", layout="wide")
st.title("PDF 문서 기반 챗봇")

# 세션 상태 초기화
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "displayed_images" not in st.session_state:
    st.session_state.displayed_images = []  # 표시된 이미지 기록 저장

# Azure OpenAI 클라이언트 생성
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# 벡터 저장소를 저장할 디렉토리 설정
VECTOR_STORE_DIR = "vector_store"
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

def encode_image_to_base64(image_bytes: bytes) -> str:
    """미지를 base64로 인코딩"""
    return base64.b64encode(image_bytes).decode('utf-8')

def optimize_image(image_data: bytes, max_size: int = 400) -> bytes:
    """이미지 최적화 개선"""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # 이미지 크기 제한
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 메모리 사용량 최적화
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=60, optimize=True)
        return output.getvalue()
    except Exception as e:
        st.warning(f"이미지 최적화 중 오류: {str(e)}")
        return image_data

def time_function(func_name: str):
    """함수 실행 시간을 측정"""
    start_time = time.time()
    def get_time():
        return time.time() - start_time
    return get_time

def process_text(reader):
    """텍스트 처리"""
    get_time = time_function("텍스트 처리")
    contents = []
    for page in reader.pages:
        text = page.extract_text()
        contents.append((text, []))
    return contents, get_time()

def process_images(reader):
    """이미지 처리"""
    get_time = time_function("이미지 처리")
    contents = []
    for page in reader.pages:
        images = []
        if '/Resources' in page and '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            images = process_images_parallel(page, xObject)
        contents.append(("", images))
    return contents, get_time()

def extract_text_from_image(image):
    """이미지에서 텍스트 추출 (OCR)"""
    try:
        text = pytesseract.image_to_string(image, lang='kor+eng')  # 한글+영어 지원
        return text.strip()
    except Exception as e:
        st.warning(f"OCR 처리 중 오류: {str(e)}")
        return ""

def extract_text_with_azure_vision(image_bytes: bytes) -> str:
    """Azure Computer Vision을 사용하여 이미지에서 텍스트 추출"""
    try:
        # Computer Vision 클라이언트 생성
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from msrest.authentication import CognitiveServicesCredentials
        
        computervision_client = ComputerVisionClient(
            AZURE_VISION_ENDPOINT,
            CognitiveServicesCredentials(AZURE_VISION_KEY)
        )
        
        # 이미지에서 텍스트 추출
        response = computervision_client.read_in_stream(image_bytes, raw=True)
        operation_location = response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        
        # 결과 대기
        import time
        while True:
            result = computervision_client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)
        
        # 텍스트 추출
        text = []
        if result.status == "succeeded":
            for text_result in result.analyze_result.read_results:
                for line in text_result.lines:
                    text.append(line.text)
        
        return "\n".join(text)
        
    except Exception as e:
        st.warning(f"Azure Vision OCR 처리 중 오류: {str(e)}")
        return ""

def process_pdf(uploaded_file, process_images=False):
    """PDF 처리 - 이미지 처리 옵션 적용"""
    try:
        start_total_time = time.time()
        pdf_bytes = uploaded_file.read()
        contents = []
        
        with st.spinner('PDF 처리 중...'):
            # PDF 로드
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            st.info(f"총 {total_pages}페이지를 처리합니다.")
            
            text_start_time = time.time()
            has_content = False
            
            progress_container = st.empty()
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                text = page.get_text().strip()
                images = []
                
                # 페스트가 있는 경우
                if text:
                    has_content = True
                    progress_container.info(f"페이지 {page_num + 1}: {len(text)} 글자 발견")
                
                # 이미지 처리 옵션이 켜져 있을 때만 이미지 처리
                if process_images:
                    # 페이지를 이미지로 변환
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("jpeg")
                    
                    # 이미지 최적화
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                        img_data = img_byte_arr.getvalue()
                        img_data = optimize_image(img_data, max_size=1200)
                        
                        img_base64 = encode_image_to_base64(img_data)
                        images.append(img_base64)
                        has_content = True
                        progress_container.info(f"페이지 {page_num + 1}: 이미지 추출 완료")
                    except Exception as e:
                        st.warning(f"페이지 {page_num + 1} 이미지 처리 중 오류: {str(e)}")
                
                contents.append((text, images))
                
                # 진행률 표시
                progress_text = f"페이지 {page_num + 1}/{total_pages} 처리 중..."
                st.progress((page_num + 1) / total_pages, text=progress_text)
            
            pdf_document.close()
            text_time = time.time() - text_start_time
            
            if not has_content:
                st.error("PDF에서 내용을 추출할 수 없습니다.")
                return None
            
            total_time = time.time() - start_total_time
            
            # 처리 시간 정보를 세션 상태에 저장
            st.session_state.processing_times = {
                "text": round(text_time, 2),
                "image": 0,
                "total": round(total_time, 2)
            }
            
            # 처리 결과 요약
            summary = f"""
            PDF 처리 완료:
            - 총 {total_pages}페이지
            - 처리 시간: {total_time:.2f}초
            """
            if process_images:
                summary += f"\n- 이미지 추출: {total_pages}장"
            
            st.success(summary)
            
            return contents
            
    except Exception as e:
        import traceback
        st.error(f"PDF 처리 중 오류 발생: {str(e)}")
        st.error(f"상세 오류 정보:\n```\n{traceback.format_exc()}\n```")
        return None

def analyze_image_content(image_base64: str) -> str:
    """GPT-4 Turbo를 사용하여 이미지 내용 분석"""
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,  # 기존 gpt-4o 모델 용
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 이미지에 무엇을 볼 수 있는지 자세히 설명해주세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"이미지 분석 중 오류 발생: {str(e)}")
        return ""

def create_vector_store(contents):
    """벡터 저장소 생성 - 이미지 처리 옵션 적용"""
    try:
        documents = []
        with st.spinner('문서 처리 중...'):
            start_total_time = time.time()
            image_start_time = time.time()
            has_content = False
            
            progress_bar = st.progress(0)
            total_items = len(contents)
            
            for idx, (text, images) in enumerate(contents):
                try:
                    page_content = ""
                    
                    # 텍스트가 있는 경우 처리
                    if text and text.strip():
                        page_content = text.strip()
                        has_content = True
                    
                    # 이미지가 있는 경우에만 이미지 분석 수행
                    if images:
                        image_descriptions = []
                        for img_idx, image_base64 in enumerate(images):
                            description = analyze_image_content(image_base64)
                            if description:
                                image_descriptions.append(f"Image {img_idx + 1}: {description}")
                                has_content = True
                        
                        if image_descriptions:
                            page_content += "\n\n" if page_content else ""
                            page_content += "Image Descriptions:\n" + "\n".join(image_descriptions)
                    
                    # 페이지에 내용이 있는 경우만 문서 추가
                    if page_content:
                        metadata = {
                            "page": idx,
                            "images": images if images else []
                        }
                        documents.append(Document(page_content=page_content, metadata=metadata))
                
                except Exception as e:
                    st.warning(f"페이지 {idx + 1} 처리 중 오류: {str(e)}")
                
                progress_bar.progress((idx + 1) / total_items)
            
            if not has_content:
                st.error("처리할 수 있는 내용이 없습니다.")
                return None
            
            image_time = time.time() - image_start_time
            
            # 임베딩 생성
            try:
                embeddings = AzureOpenAIEmbeddings(
                    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
                    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
                
                vector_store = FAISS.from_documents(documents, embeddings)
                
                total_time = time.time() - start_total_time
                
                # 처리 시간 업데이트
                if "processing_times" in st.session_state:
                    times = st.session_state.processing_times
                    st.session_state.processing_times.update({
                        "image": round(image_time, 2) if any(doc.metadata.get("images") for doc in documents) else 0,
                        "total": round(times["total"] + total_time, 2)
                    })
                
                st.success("벡터 저장소 생성 완료")
                return vector_store
                
            except Exception as e:
                st.error(f"벡터 저장소 생성 중 오류: {str(e)}")
                return None
            
    except Exception as e:
        st.error(f"문서 처리 중 오류 발생: {str(e)}")
        return None

def initialize_conversation(vector_store):
    llm = AzureChatOpenAI(
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=0,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation

def save_vector_store(vector_store, file_name):
    """벡터 저장소를 파일로 저장"""
    save_path = os.path.join(VECTOR_STORE_DIR, file_name)
    vector_store.save_local(save_path)
    return save_path

def load_vector_store(file_name, embeddings):
    """저장된 벡터 저장소 로드"""
    load_path = os.path.join(VECTOR_STORE_DIR, file_name)
    if os.path.exists(load_path):
        return FAISS.load_local(load_path, embeddings)
    return None

def delete_vector_store(file_name):
    """벡터 저장소 삭제"""
    path = os.path.join(VECTOR_STORE_DIR, file_name)
    if os.path.exists(path):
        shutil.rmtree(path)

def process_images_parallel(page, xObject):
    """이미지 병렬 처리"""
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                futures.append(executor.submit(process_single_image, xObject[obj]))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                images.append(result)
    return images

def process_single_image(image):
    """단일 이미지 처리"""
    try:
        if '/Filter' in image:
            if image['/Filter'] == '/DCTDecode':
                img_data = image._data
            else:
                size = (image['/Width'], image['/Height'])
                img_data = Image.frombytes('RGB', size, image._data)
                img_byte_arr = io.BytesIO()
                img_data.save(img_byte_arr, format='JPEG', quality=60)  # 품질 낮
                img_data = img_byte_arr.getvalue()
            
            # 이미지 크기 제한
            img_data = optimize_image(img_data, max_size=400)
            return encode_image_to_base64(img_data)
    except Exception as e:
        st.warning(f"이미지 처리 중 오류 발생: {str(e)}")
        return None

def analyze_images_batch(images, batch_size=4):
    """이미지 배치 분"""
    descriptions = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(analyze_image_content, img) for img in batch]
            for future in concurrent.futures.as_completed(futures):
                description = future.result()
                if description:
                    descriptions.append(description)
    return descriptions

def analyze_images_in_batch(documents, batch_size=4):
    """이미지 배치 분석"""
    docs_with_images = [doc for doc in documents if doc.metadata.get("images")]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(0, len(docs_with_images), batch_size):
            batch = docs_with_images[i:i + batch_size]
            futures = []
            
            for doc in batch:
                for image_base64 in doc.metadata["images"]:
                    futures.append(
                        executor.submit(analyze_image_content, image_base64, doc)
                    )
            
            concurrent.futures.wait(futures)

# 캐시된 함수들 먼저 정의
@st.cache_data(show_spinner=False)
def process_and_cache_image(image_data, image_key, file_hash):
    """이미지 처리 결과 캐시"""
    try:
        img_data = optimize_image(image_data)
        base64_img = encode_image_to_base64(img_data)
        return base64_img
    except Exception as e:
        st.warning(f"이미지 캐시 처리 중 오류: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def analyze_and_cache_image(image_base64, image_key, file_hash):
    """이미지 분석 결과 캐시"""
    return analyze_image_content(image_base64)

@st.cache_data(show_spinner=False)
def get_file_hash(file_content: bytes) -> str:
    """파일 용 해시값 생성"""
    return hashlib.md5(file_content).hexdigest()

# 그 다음 캐시 관리 함수 정의
def clear_cache_for_file(file_name: str):
    """특정 파일에 대한 캐시 초기화"""
    try:
        # 이미��� 처리 캐시 초기화
        process_and_cache_image.clear()
        # 이미지 분석 캐시 초기화
        analyze_and_cache_image.clear()
        # 기타 캐시 초기화
        st.cache_data.clear()
    except Exception as e:
        st.warning(f"캐시 초기화 중 오류 발생: {str(e)}")

# 현재 처리 중인 파일 정보
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

def get_memory_usage():
    """시스템 메모리 사용량 조"""
    import psutil
    process = psutil.Process()
    
    # 프로세스 메모리 사용량
    process_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
    
    # 시스템 메모리 정보
    system_memory = psutil.virtual_memory()
    total_memory = system_memory.total / 1024 / 1024  # MB 단위
    available_memory = system_memory.available / 1024 / 1024  # MB 단위
    used_memory = system_memory.used / 1024 / 1024  # MB 단위
    
    # 캐시 메모리 정보 (Linux 시스템의 경우)
    try:
        cached_memory = system_memory.cached / 1024 / 1024  # MB 단위
    except:
        cached_memory = 0
    
    return {
        "process": round(process_memory, 2),
        "total": round(total_memory, 2),
        "available": round(available_memory, 2),
        "used": round(used_memory, 2),
        "cached": round(cached_memory, 2)
    }

# 함수들을 파일 상단으로 이동
def display_processing_times():
    """처리 시간 정보 표시"""
    if "processing_times" in st.session_state:
        times = st.session_state.processing_times
        st.sidebar.subheader("처리 시간")
        
        # 처리 시간 표시
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("텍스트 처리", f"{times['text']}초")
            st.metric("이미지 처리", f"{times['image']}초")
        with col2:
            st.metric("총 처리 시간", f"{times['total']}초")
        
        # 처리 시간 차트
        if times['image'] > 0:  # 이미지 처리가 수행된 경우에만
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Bar(name='처리 시간',
                      x=['텍스트', '이미지', '총 시간'],
                      y=[times['text'], times['image'], times['total']],
                      text=[f"{t:.2f}s" for t in [times['text'], times['image'], times['total']]],
                      textposition='auto')
            ])
            fig.update_layout(
                title="처리 시간 분석",
                height=200,
                showlegend=False
            )
            st.sidebar.plotly_chart(fig, use_container_width=True)

# 사이드바: PDF 업로드 부분 수정
with st.sidebar:
    st.header("PDF 문서 선택")
    
    # PDF 소스 선택
    pdf_source = st.radio(
        "PDF 소스 선택",
        ["기본 문서 (china_music.pdf)", "파일 업로드"]
    )
    
    if pdf_source == "파일 업로드":
        uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
        if uploaded_file:
            # 파일 해시 계산
            file_content = uploaded_file.read()
            file_hash = get_file_hash(file_content)
            uploaded_file.seek(0)  # 파일 포인터 리셋
            
            # 새로운 파일이 업로된 경우 캐시 초기화
            if st.session_state.current_file_hash != file_hash:
                clear_cache_for_file(uploaded_file.name)
                st.session_state.current_file_hash = file_hash
    else:
        # 기본 PDF 파일 경로 확인
        default_pdf_path = "./china_music.pdf"
        if not os.path.exists(default_pdf_path):
            st.error("기본 PDF 파일을 찾을 수 없습니다.")
            uploaded_file = None
        else:
            with open(default_pdf_path, "rb") as f:
                file_content = f.read()
                file_hash = get_file_hash(file_content)
                
                # 기본 문서로 전환 시 캐시 초기화
                if st.session_state.current_file_hash != file_hash:
                    clear_cache_for_file("china_music.pdf")
                    st.session_state.current_file_hash = file_hash
            
            uploaded_file = open(default_pdf_path, "rb")
            st.success("기본 PDF 파일이 로드되었습니다.")
    
    process_images = st.checkbox("이미지 처리 포함", value=True)
    save_vector = st.checkbox("벡터 저장소 저장", value=False, help="처리된 벡터 저장소를 파일로 저장하여 다음에 재사용할 수 있습니다.")
    
    if uploaded_file and st.button("처리 시작"):
        try:
            # 파일명으로 벡터 저장소 식별자 생성
            if pdf_source == "파일 업로드":
                store_name = uploaded_file.name.replace('.pdf', '')
            else:
                store_name = "china_music"
            
            vector_store = None
            if save_vector:
                # 기 벡터 저장소 확인
                embeddings = AzureOpenAIEmbeddings(
                    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
                    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
                vector_store = load_vector_store(store_name, embeddings)
            
            # 벡터 저장소가 없으면 새로 생성
            if vector_store is None:
                contents = process_pdf(uploaded_file, process_images)
                if contents:
                    vector_store = create_vector_store(contents)
                    if vector_store and save_vector:
                        # 벡터 저장소 저장
                        save_vector_store(vector_store, store_name)
                        st.success(f"벡터 저장소가 '{store_name}' 이름으로 저장되었습니다.")
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.conversation = initialize_conversation(vector_store)
                st.success("PDF 처리가 완료되었습니다!")
                st.session_state.chat_history = []
                
                # 처리 시간 표시
                display_processing_times()
                
        finally:
            if pdf_source == "기본 문서 (china_music.pdf)" and uploaded_file:
                uploaded_file.close()

    # 저장된 벡터 저장소가 있을 때만 초기화 버튼 표시
    if os.path.exists(os.path.join(VECTOR_STORE_DIR, "china_music")) or \
       (pdf_source == "파일 업로드" and uploaded_file and 
        os.path.exists(os.path.join(VECTOR_STORE_DIR, uploaded_file.name.replace('.pdf', '')))):
        if st.button("벡터 저장소 초기화"):
            if pdf_source == "파일 업로드" and uploaded_file:
                store_name = uploaded_file.name.replace('.pdf', '')
            else:
                store_name = "china_music"
            delete_vector_store(store_name)
            st.session_state.vector_store = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.success("벡터 저장소가 초기화되었습니다.")

    if st.sidebar.checkbox("디버그 정보 표시"):
        import psutil
        process = psutil.Process()
        st.sidebar.info(f"메모리 사용량: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        st.sidebar.info(f"현재 파일: {st.session_state.current_file_hash}")

    if st.checkbox("시스템 모니터링", value=False):
        memory_info = get_memory_usage()
        
        st.subheader("메모리 사용량")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("프로세스", f"{memory_info['process']} MB")
            st.metric("사용 가능", f"{memory_info['available']} MB")
            st.metric("캐시", f"{memory_info['cached']} MB")
        
        with col2:
            st.metric("전체", f"{memory_info['total']} MB")
            st.metric("사용 중", f"{memory_info['used']} MB")
        
        # 메모리 사용량 차트
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Pie(
            values=[memory_info['available'], memory_info['used'], memory_info['cached']],
            labels=['사용 가능', '사용 중', '캐시'],
            hole=.3
        ))
        fig.update_layout(
            title="메모리 사용량 분포",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 캐시된 이미지 수
        cached_items = len(st.session_state.keys())
        st.metric("캐시된 항목 수", cached_items)
        
        if st.button("캐시 비우기"):
            st.cache_data.clear()
            st.experimental_rerun()

# 자동 메모리 모니터링 (30초마다 갱신)
if "monitor_memory" not in st.session_state:
    st.session_state.monitor_memory = False

if st.session_state.monitor_memory:
    import time
    
    placeholder = st.empty()
    while True:
        with placeholder.container():
            memory_info = get_memory_usage()
            st.metric("현재 메모리 사용량", f"{memory_info['process']} MB")
        time.sleep(30)

# 메인 화면: 채팅 인터페이스 부분 수정
if st.session_state.conversation is None:
    st.info("👈 시작하려면 PDF 파일을 업로드하세요.")
else:
    chat_container = st.container()
    with chat_container:
        # 채팅 기록 표시
        for message in st.session_state.chat_history:
            if isinstance(message, tuple):
                user_message, bot_message = message
                with st.chat_message("user"):
                    st.write(user_message)
                with st.chat_message("assistant"):
                    st.write(bot_message)
                    
        # 저장된 이미지 표시
        for image_data in st.session_state.displayed_images:
            with st.chat_message("assistant"):
                st.info(image_data["caption"])
                image = Image.open(io.BytesIO(base64.b64decode(image_data["image"])))
                st.image(image, caption=image_data["index"], use_column_width=True)
    
    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = st.session_state.conversation({"question": user_input})
                st.write(response["answer"])
                
                if process_images:
                    displayed_images_count = 0  # 표시된 이미지 수 추적
                    for doc in response.get("source_documents", []):
                        if doc.metadata.get("images") and displayed_images_count < 2:  # 최대 2개까지만 처리
                            st.info("이 답변과 관련된 이미지:")
                            for idx, image_base64 in enumerate(doc.metadata["images"]):
                                if displayed_images_count >= 2:  # 2개 이상이면 중단
                                    break
                                try:
                                    image_bytes = base64.b64decode(image_base64)
                                    image = Image.open(io.BytesIO(image_bytes))
                                    st.image(image, caption=f"관련 이미지 {displayed_images_count + 1}", use_column_width=True)
                                    
                                    # 이미지 정보를 세션 상태에 저장
                                    st.session_state.displayed_images.append({
                                        "image": image_base64,
                                        "caption": "이 답변과 관련된 이미지:",
                                        "index": f"관련 이미지 {displayed_images_count + 1}"
                                    })
                                    displayed_images_count += 1
                                except Exception as e:
                                    st.warning(f"이미지 표��� 중 오류 발생: {str(e)}")
                                    
                    if displayed_images_count == 2:  # 2개가 표시된 경우 메시지 표시
                        st.info("(최대 2개의 관련 이미지만 표시됩니다)")
        
        # 채팅 기록 저장
        st.session_state.chat_history.append((user_input, response["answer"]))

def create_optimized_vector_store(documents, embeddings):
    """최적화된 벡터 저장소 생성"""
    # 문서를 배치로 처리
    batch_size = 32
    vector_store = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vector_store.merge_from(batch_store)
    
    return vector_store

def create_embeddings_batch(texts: List[str], embeddings: AzureOpenAIEmbeddings, batch_size: int = 16):
    """배치 단위로 임베딩 생성"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

def optimize_document_content(documents: List[Document]):
    """문서 내용 최적화"""
    for doc in documents:
        # 텍스트 길이 제한
        if len(doc.page_content) > 1000:
            doc.page_content = doc.page_content[:1000]
        
        # 이미지 크기 제한
        if doc.metadata.get("images"):
            doc.metadata["images"] = [
                optimize_image_base64(img) for img in doc.metadata["images"]
            ]
    return documents

def create_documents_parallel(contents):
    """병렬로 문서 생성"""
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    def process_content(idx, text, images):
        if not text.strip():
            return []
        
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "page": idx,
                    "chunk": chunk_num,
                    "images": images if images else []
                }
            )
            for chunk_num, chunk in enumerate(chunks)
        ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_content, idx, text, images)
            for idx, (text, images) in enumerate(contents)
        ]
        
        with st.progress(0) as progress_bar:
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                documents.extend(future.result())
                progress_bar.progress((idx + 1) / len(futures))
    
    return documents
