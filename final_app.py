# -*- coding: utf-8 -*-
"""
Chative Jobs — Gradio RAG + Streaming (팀 템플릿 유지 버전)
- UI: 랜딩 → PDF 미리보기 + 우측 챗 (그대로 유지)
- 모델: HF(Base) + LoRA 스트리밍
- 내부 RAG: 업로드 PDF → 문단 → SBERT(384) → 세션 FAISS
- 외부 RAG: ./faiss_index/{faiss.index, store.jsonl} 존재 시 사용 (OpenAI 임베딩 쿼리)
필요:
  pip install -U gradio transformers peft torch faiss-cpu sentence-transformers PyMuPDF pillow openai
"""

import os, re, json, threading, base64, time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from PIL import Image
import numpy as np
import faiss
import fitz  # PyMuPDF
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ========= 🔐 토큰 (하드코딩 or 환경변수) =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  
HF_TOKEN       = os.getenv("HF_TOKEN", "") 

# ========= HF 모델 =========
BASE        = "K-intelligence/Midm-2.0-Mini-Instruct"
ADAPTER_DIR = "monkey777/midm2-mini-lora-finetune"

# ========= 경로 =========
BASE_DIR = Path(".")
FAISS_DIR = BASE_DIR / "faiss_index"
CONTENT_INDEX_FP = FAISS_DIR / "faiss.index"
CONTENT_META_FP  = FAISS_DIR / "store.jsonl"

# ========= 하이퍼파라미터 =========
TOPK_INTERNAL = 6
TOPK_CONTENT  = 6
MAX_INTERNAL_CHARS = 1200
MAX_CONTENT_CHARS  = 1500

SYSTEM_PROMPT = (
    "너는 최고의 마케팅 전략가 겸 카피라이터다. "
    "제공된 내부/외부 컨텍스트를 **복붙하지 말고 요약/가공**해 한국어로 간결하고 명료하게 답하라. "
    "컨텍스트가 부족하면 모델 지식으로 보강하되, 근거가 약하면 '추정:'으로 구분하라. "
    "가능하면 다음 구조를 따르라:\n"
    "## RAG 요약(5~8개)\n"
    "## 전략(세그먼트별: 페인/니즈 → 메시지 → 채널/형식)\n"
    "## 실행안(3~5개 실험: 가설·지표·리스크 + 샘플 카피)\n"
    "## 측정(KPI와 수집 방법)"
)

OPENAI_EMBED_MODEL = "text-embedding-3-large"  # 외부 인덱스가 해당 차원으로 생성되었다고 가정(보통 3072)

# ========= 로컬 임베딩(SBERT 384) =========
_local_embedder: Optional[SentenceTransformer] = None
def get_embedder():
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _local_embedder

def emb_texts_local(texts: List[str]) -> np.ndarray:
    embs = get_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype("float32")

def emb_query_local(q: str) -> np.ndarray:
    v = get_embedder().encode([q], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return v.astype("float32")

# ========= OpenAI 임베딩(외부 인덱스) =========
def embed_query_openai(q: str) -> Optional[np.ndarray]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=15)
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
        v = np.array([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(v)
        return v
    except Exception as e:
        print(f"[WARN] OpenAI embed failed: {e}")
        return None

# ========= PDF → 텍스트/문단 =========
PARA_SPLIT_RE = re.compile(r"(?:\r?\n\s*){2,}")

def pdf_to_text(pdf_bytes: bytes) -> str:
    pages=[]
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            words = page.get_text("words")
            if words:
                words.sort(key=lambda w: (w[5], w[6], w[7], w[0]))
                lines_map={}
                for x0,y0,x1,y1,txt,bno,lno,wno in words:
                    if not txt: continue
                    lines_map.setdefault((bno,lno), []).append(txt)
                lines = [" ".join(tokens) for _, tokens in sorted(lines_map.items(), key=lambda k:k[0])]
                pages.append("\n".join(lines))
            else:
                pages.append(page.get_text("text") or "")
    return "\n\n".join(pages)

def _letters_digits(s: str):
    letters = sum(1 for ch in s if ('A'<=ch<='Z') or ('a'<=ch<='z') or ('\uAC00'<=ch<='\uD7A3'))
    digits  = sum(1 for ch in s if ch.isdigit())
    return letters, digits

def text_to_paragraphs(text: str, min_chars=120, drop_digit_heavy=True) -> List[str]:
    text = text.replace("\r\n","\n")
    parts = [re.sub(r"[ \t]*\n[ \t]*", " ", p.strip()) for p in PARA_SPLIT_RE.split(text)]
    out=[]
    for p in parts:
        if len(p) < min_chars: continue
        if drop_digit_heavy:
            L,D = _letters_digits(p)
            if D > L: continue
        out.append(re.sub(r"\s+", " ", p).strip())
    return out

# ========= 외부 메타 =========
def load_meta(fp: Path) -> List[Dict]:
    assert fp.exists(), f"Meta not found: {fp}"
    if fp.suffix.lower() == ".jsonl":
        out=[]
        with fp.open("r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                obj=json.loads(ln)
                out.append({"id": obj.get("id",""), "text": obj.get("document") or obj.get("text","")})
        return out
    data=json.loads(fp.read_text(encoding="utf-8"))
    if isinstance(data, list): return data
    if "records" in data:
        return [{"id": r["id"], "text": r.get("document") or r.get("text","")} for r in data["records"]]
    return data

# ========= 검색 유틸 =========
def faiss_search(index, meta, qv: np.ndarray, k: int):
    if index is None or meta is None or qv is None: return []
    D,I = index.search(qv, int(k))
    hits=[]
    for d,i in zip(D[0], I[0]):
        if i < 0 or i >= len(meta): continue
        rec = meta[i]
        hits.append({"id": rec.get("id",""), "text": rec.get("text",""), "score": float(d)})
    return hits

def limit_chars(items: List[Dict], key="text", cap=1000) -> List[str]:
    out=[]; used=0
    for it in items:
        t=(it.get(key) or "").strip()
        if not t: continue
        if used + len(t) > cap: break
        out.append(t); used += len(t)
    return out

# ========= 프롬프트 =========
def sanitize_user_text(s: str) -> str:
    if not s: return s
    s = s.replace("\uFFFD", "").replace("\ufeff","").replace("\uFEFF","")
    return s.strip()

def build_rag_block(internal_ctx: List[str], content_ctx: List[str]) -> str:
    def pack(lines: List[str], cap_each: int, max_lines: int, bullet: str) -> List[str]:
        out=[]
        for t in lines:
            t=(t or "").strip()
            if not t: continue
            if len(t)>cap_each: t=t[:cap_each].rstrip()+"…"
            out.append(f"{bullet} {t}")
            if len(out)>=max_lines: break
        return out
    internal_block = "\n".join(pack(internal_ctx, 300, 6, "-")) if internal_ctx else "(내부 컨텍스트 없음)"
    content_block  = "\n".join(pack(content_ctx,  280, 6, "-")) if content_ctx  else "(외부 컨텍스트 없음)"
    return f"[내부 데이터(우선)]\n{internal_block}\n\n[외부 연구(보강)]\n{content_block}\n\n※ 컨텍스트가 부족하면 모델 지식으로 보완하라."

def build_prompt_from_messages(messages: List[Dict], rag_block: str, tok) -> str:
    # 마지막 user 메시지에 RAG 블록을 붙여준다
    msgs = [{"role":"system","content": SYSTEM_PROMPT}]
    for m in messages[:-1]:
        if m.get("role") in ("user","assistant") and (m.get("content") or "").strip():
            msgs.append({"role": m["role"], "content": m["content"]})
    last = messages[-1]
    merged_user = f"{sanitize_user_text(last.get('content',''))}\n\n{rag_block}"
    msgs.append({"role":"user","content": merged_user})

    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    compiled=[]
    for m in msgs:
        compiled.append(f"## {m['role'].upper()}\n{m['content']}")
    compiled.append("## ASSISTANT\n")
    return "\n\n".join(compiled)

# ========= 모델 로드 =========
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, token=(HF_TOKEN or None))
base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype="auto", device_map="auto", trust_remote_code=True, token=(HF_TOKEN or None)
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR, token=(HF_TOKEN or None)).eval()
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

# ========= 외부(마케팅) 인덱스 =========
CONTENT_INDEX = None
CONTENT_META  = None
EXT_DIM = None
CONTENT_READY = False
CONTENT_ERR = None
try:
    if CONTENT_INDEX_FP.exists() and CONTENT_META_FP.exists():
        CONTENT_INDEX = faiss.read_index(str(CONTENT_INDEX_FP))
        CONTENT_META  = load_meta(CONTENT_META_FP)
        EXT_DIM = CONTENT_INDEX.d
        CONTENT_READY = True
        print(f"[RAG] 외부 인덱스 로드됨. dim={EXT_DIM}, docs={len(CONTENT_META)}")
    else:
        CONTENT_ERR = "외부 인덱스/메타 파일이 없습니다."
except Exception as e:
    CONTENT_ERR = f"외부 인덱스 로드 오류: {e}"
    CONTENT_READY = False
    print(f"[WARN] {CONTENT_ERR}")

# ========= 내부(세션) 인덱스 =========
INTERNAL_INDEX: Optional[faiss.IndexFlatIP] = None
INTERNAL_META: List[Dict] = []

def build_internal_index_from_files(files: List) -> str:
    """업로드된 PDF들로 세션 인덱스 생성(시간 측정)"""
    global INTERNAL_INDEX, INTERNAL_META
    t0 = time.time()
    if not files:
        INTERNAL_INDEX, INTERNAL_META = None, []
        return "📎 업로드된 PDF가 없습니다."

    all_paras=[]
    try:
        for pf in files:
            raw = pf.read() if hasattr(pf, "read") else open(pf.name, "rb").read()
            txt = pdf_to_text(raw)
            all_paras += text_to_paragraphs(txt, min_chars=120, drop_digit_heavy=True)
        if not all_paras:
            INTERNAL_INDEX, INTERNAL_META = None, []
            return "⚠️ 문단 추출 결과가 비었습니다."
        vecs = emb_texts_local(all_paras)  # 384차원
        idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
        meta = [{"id": f"internal::{i:06d}", "text": t} for i,t in enumerate(all_paras)]
        INTERNAL_INDEX, INTERNAL_META = idx, meta
        dt = time.time() - t0
        return f"✅ 내부 인덱스 구축 완료 (문단 {len(all_paras)}, dim={vecs.shape[1]}) — {dt:.2f}s"
    except Exception as e:
        INTERNAL_INDEX, INTERNAL_META = None, []
        return f"❌ 내부 인덱스 오류: {e}"

# ========= 생성(스트리밍) =========
def stream_answer(messages: List[Dict],
                  k_int: int, k_ext: int,
                  cap_int: int, cap_ext: int,
                  max_new_tokens: int, temperature: float, top_p: float,
                  repetition_penalty: float):
    """messages(OpenAI 스타일)를 받아 스트리밍 생성"""
    # 1) 쿼리 텍스트
    user_text = (messages[-1].get("content") or "").strip()
    if not user_text:
        yield ""
        return

    # 2) RAG 검색
    internal_ctx=[]; content_ctx=[]
    try:
        # 내부: 로컬(384)
        if INTERNAL_INDEX is not None and INTERNAL_META and int(k_int) > 0:
            qv_local = emb_query_local(user_text)
            internal_hits = faiss_search(INTERNAL_INDEX, INTERNAL_META, qv_local, int(k_int))
            internal_ctx = limit_chars(internal_hits, "text", int(cap_int))

        # 외부: OpenAI 쿼리 → 차원검증
        if CONTENT_READY and CONTENT_INDEX is not None and CONTENT_META is not None and int(k_ext) > 0:
            qv_ext = embed_query_openai(user_text)
            if qv_ext is not None and qv_ext.shape[1] == CONTENT_INDEX.d:
                content_hits = faiss_search(CONTENT_INDEX, CONTENT_META, qv_ext, int(k_ext))
                content_ctx  = limit_chars(content_hits, "text", int(cap_ext))
    except Exception as e:
        print(f"[WARN] RAG search failed: {e}")

    # 3) 프롬프트 구성
    rag_block = build_rag_block(internal_ctx, content_ctx)
    prompt = build_prompt_from_messages(messages, rag_block, tok)

    # 4) 스트리머
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )

    th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    th.start()
    partial = ""
    for new_text in streamer:
        partial += new_text
        yield partial
    th.join()

# ========= PDF 렌더링(좌측 미리보기) =========
def render_pdf_to_images_html(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        image_htmls = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=220)
            img_bytes = pix.tobytes("png")
            encoded = base64.b64encode(img_bytes).decode('utf-8')
            image_htmls.append(
                f'<img src="data:image/png;base64,{encoded}" alt="Page {i+1}" '
                f'style="max-width: 100%; margin-bottom: 10px; border-radius: 8px;"/>'
            )
        return "".join(image_htmls)
    except Exception as e:
        return f"<p style='color: red;'>PDF 렌더링 오류: {e}</p>"

# ========= 아바타 =========
if not os.path.exists("assets"):
    os.makedirs("assets")
avatar_path = "assets/steve_jobs.jpg"
if not os.path.exists(avatar_path):
    img = Image.new('RGB', (100, 100), color = '#222222')
    img.save(avatar_path)

# ========= CSS (팀 템플릿 유지) =========
app_css = """
@import url('https://cdn.jsdelivr.net/gh/sun-typeface/SUIT/fonts/variable/woff2/SUIT-Variable.css');
body, #app-container { font-family: 'SUIT Variable', sans-serif; background-color: #000000 !important; color: #FFFFFF !important; }
.gradio-container { background-color: #000000 !important; }
#landing-page { padding: 0 !important; margin: 0 !important; }
.section { min-height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 2rem; box-sizing: border-box; }
.section h1 { font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; color: #f5f5f7;}
.section h2 { font-size: 2.5rem; font-weight: 700; margin-bottom: 1.5rem; color: #f5f5f7;}
.section p { font-size: 1rem; max-width: 700px; line-height: 1.6; color: #a1a1a6; }
.upload-section { background-color: #1d1d1f; border-radius: 20px; margin: 2rem auto; max-width: 800px; }
#pdf-uploader > label { display: none !important; }
#pdf-uploader { background: transparent !important; border: 2px dashed #48484a !important; border-radius: 15px; padding: 2rem 1rem; min-height: 150px; }
#pdf-uploader .file-drop { background: transparent !important; border: none !important; color: #a1a1a6 !important;
    display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 1rem; }
#pdf-uploader .file-drop svg { color: #a1a1a6 !important; }
#pdf-uploader .file-drop button { color: #a1a1a6 !important; background: none !important; text-decoration: underline; }
#pdf-uploader .file-list { background: transparent !important; border: none !important; box-shadow: none !important; }
#pdf-uploader .file-list .file { background-color: transparent !important; color: #a1a1a6 !important; border: none !important; border-radius: 8px !important; }
#pdf-uploader .file-list .file .progress-bar { background-color: transparent !important; }
#pdf-uploader .file-list .file .progress-bar .progress { background-color: #007aff !important; height: 4px !important; }
#main-app-interface { background-color: #F0F2F6 !important; color: #000 !important; border-radius: 20px; padding: 1rem;}
#main-app-interface .gradio-container { background-color: transparent !important; }
#pdf-display-container h3 { color: #1d1d1f; }
#chatbot .user { background-color: #007aff !important; border-radius: 18px 18px 5px 18px !important; }
#chatbot .user .prose * { color: #FFFFFF !important; font-weight: bold; }
#chatbot .bot { background-color: #e5e5ea !important; color: #000 !important; border-radius: 18px 18px 18px 5px !important; }
#chatbot .message { border: none !important; box-shadow: none !important; }
#main-app-interface .gr-button { background-color: #007aff; color: white; }
#main-app-interface .gr-button.secondary { background-color: #e5e5ea; color: #000; }
#pdf-display-container { background-color: white; border-radius: 12px; padding: 1rem; overflow-y: auto; height: 93vh; }
#chatbot-container { background-color: #ffffff; border-radius: 12px; padding: 1rem; height: 93vh; display: flex; flex-direction: column; }
#chatbot { flex-grow: 1; min-height: 65vh; }
#chatbot .message p { font-size: 14px !important; line-height: 1.6 !important; }
#chatbot .message .prose * { font-size: 14px !important; line-height: 1.6 !important; }
#chatbot-container textarea { font-size: 14px !important; }
"""

# ========= Gradio UI =========
with gr.Blocks(css=app_css, title="Chative Jobs") as demo:

    # States
    st_messages = gr.State([])           # [{role, content}]
    st_uploaded_pdf_path = gr.State(None)

    # --- Landing ---
    with gr.Column(elem_id="landing-page") as landing_page:
        gr.HTML("""
        <div class="section">
            <h1>Chative Jobs</h1>
            <p style="font-size: 1.5rem; color: #f5f5f7;">마케팅 전략, 다시 생각할 시간.</p>
        </div>
        <div class="section">
            <h2>전략의 본질. 잡스의 언어.</h2>
            <p>파인튜닝으로 완성된 잡스의 뉴럴 엔진.<br>
            수백만 단어에서 추출한 그의 철학. 수십 번의 키노트에서 증명된 그의 화법.<br>
            수십 통의 이메일에서 드러난 그의 본질.</p>
        </div>
        <div class="section">
            <h2>궁극의 브랜딩 피드백 시스템.</h2>
            <p>어떤 기획서든, 그 핵심을 꿰뚫는 통찰력.<br>
            여기에 업계 사상 가장 정직한 피드백까지.</p>
        </div>
        <div class="section">
            <h2>당신의 비전. 더 단단하게.</h2>
            <p>새롭게 선보이는 Chative Jobs는 당신의 아이디어를 위한 담금질의 과정입니다.<br>
            복잡한 기능 목록을 늘어놓는 대신, 사용자에게 전할 단 하나의 가치를 찾아내죠.<br>
            소음을 걷어내고 본질만 남기는 것. 위대한 전략은 바로 거기서 시작되니까.</p>
        </div>
        <div class="section">
            <h2>단지 AI가 아니다.<br>잡스다.</h2>
            <p>단순히 정보를 요약하는 챗봇은 많습니다.<br>
            하지만 Chative Jobs는 관점을 제시합니다.<br>
            당신의 아이디어가 세상을 바꿀 수 있는지, 아니면 그저 그런 제품으로 남을 것인지.<br>
            그 차이를 만듭니다.</p>
        </div>
        <div class="section">
            <h2>이제, 당신의 차례다.</h2>
            <p>당신의 기획서를 업로드하고,<br>가장 위대한 마케터의 피드백을 경험하세요.</p>
        </div>
        """)
        pdfs = gr.Files(label="📎 PDF 파일 업로드 (복수 선택 가능)", file_types=[".pdf"], elem_id="pdf-uploader")

    # --- Main App ---
    with gr.Column(visible=False, elem_id="main-app-interface") as main_app:
        with gr.Row():
            # Left: PDF preview
            with gr.Column(scale=5):
                with gr.Column(elem_id="pdf-display-container"):
                    gr.HTML("<h3 style='text-align: center; color: #1d1d1f;'>📄 Uploaded Document</h3>")
                    pdf_viewer = gr.HTML()
                    build_log  = gr.Textbox(label="인덱스 로그", lines=3)

            # Right: Chat
            with gr.Column(scale=5):
                with gr.Column(elem_id="chatbot-container"):
                    gr.HTML("<h3 style='text-align: center; color: #1d1d1f;'>🍎 Chative Jobs Feedback</h3>")
                    chatbot = gr.Chatbot(
                        height=600,
                        elem_id="chatbot",
                        type="messages",
                        avatar_images=(None, avatar_path)
                    )
                    user_box = gr.Textbox(placeholder="질문을 입력하세요…", show_label=False)

                with gr.Row():
                    clear_btn = gr.Button("대화 지우기", variant="secondary")

                with gr.Accordion("RAG 및 생성 옵션", open=False):
                    with gr.Row():
                        with gr.Column():
                            topk_internal = gr.Slider(0, 20, value=TOPK_INTERNAL, step=1, label="TOPK_INTERNAL (내부)")
                            topk_content  = gr.Slider(0, 20, value=TOPK_CONTENT,  step=1, label="TOPK_CONTENT (외부)")
                            cap_internal  = gr.Slider(200, 4000, value=MAX_INTERNAL_CHARS, step=100, label="MAX_INTERNAL_CHARS")
                            cap_content   = gr.Slider(200, 4000, value=MAX_CONTENT_CHARS,  step=100, label="MAX_CONTENT_CHARS")
                            gr.Markdown("외부 인덱스 상태: " + ("✅ 사용 가능" if CONTENT_READY else f"❌ 미사용 — {CONTENT_ERR or '파일 없음'}"))
                        with gr.Column():
                            max_new_tokens = gr.Slider(64, 4096, value=1024, step=32, label="max_new_tokens")
                            temperature    = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                            top_p          = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                            repetition_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="repetition_penalty")

    # ----- Handlers -----

    def on_files_uploaded(files: List, messages: List):
        """업로드 → 내부 인덱스 빌드(시간표시) + UI 전환 + 첫 안내"""
        log = build_internal_index_from_files(files)

        landing_vis = gr.update(visible=False)
        main_vis    = gr.update(visible=True)

        first_path = files[0].name
        pdf_html = render_pdf_to_images_html(first_path)

        sys_msg = {"role": "assistant", "content": "문서 업로드 완료. 핵심만 뽑아 전략적으로 도울게요. 무엇이 궁금한가요?"}
        new_msgs = (messages or []) + [sys_msg]
        return landing_vis, main_vis, pdf_html, log, first_path, new_msgs, new_msgs

    def on_user_submit(user_text: str, messages: List[Dict]):
        """사용자 입력을 messages에 추가 + assistant placeholder 추가(스트리밍 표시용)"""
        if not user_text or not user_text.strip():
            return "", messages, messages
        msgs = (messages or [])
        msgs = msgs + [{"role":"user","content": user_text.strip()}]
        # 스트리밍 중 화면 갱신을 위해 미리 assistant 빈 메시지 추가
        msgs = msgs + [{"role":"assistant","content": ""}]
        return "", msgs, msgs

    def on_bot_stream(messages: List[Dict],
                      k_int, k_ext, cap_int, cap_ext,
                      max_new, temp, tp, rep_pen):
        """마지막 assistant 빈 메시지를 실시간 업데이트"""
        if not messages:
            yield messages
            return
        # 입력으로 줄 messages_for_model = 마지막 assistant 제거(즉, 마지막 user까지)
        if messages[-1]["role"] == "assistant":
            messages_for_model = messages[:-1]
        else:
            messages_for_model = messages

        stream = stream_answer(messages_for_model,
                               k_int, k_ext, cap_int, cap_ext,
                               max_new, temp, tp, rep_pen)
        partial = ""
        for chunk in stream:
            partial = chunk
            # 마지막이 assistant placeholder라고 가정하고 업데이트
            if messages[-1]["role"] != "assistant":
                messages.append({"role":"assistant","content": ""})
            messages[-1]["content"] = partial
            yield messages

    def on_clear():
        return [], gr.update(visible=True), gr.update(visible=False), None, "", ""

    # ----- Wiring -----

    # 업로드 → 전환 & 인덱스 빌드 & 초기 안내
    pdfs.upload(
        on_files_uploaded,
        inputs=[pdfs, st_messages],
        outputs=[landing_page, main_app, pdf_viewer, build_log, st_uploaded_pdf_path, chatbot, st_messages]
    )

    # 사용자 입력 → messages에 user+assistant(빈) 추가 → 스트리밍으로 assistant 채우기
    user_box.submit(
        on_user_submit,
        inputs=[user_box, st_messages],
        outputs=[user_box, st_messages, chatbot]
    ).then(
        on_bot_stream,
        inputs=[st_messages, topk_internal, topk_content, cap_internal, cap_content,
                max_new_tokens, temperature, top_p, repetition_penalty],
        outputs=[chatbot]
    )

    # 초기화
    clear_btn.click(
        on_clear,
        outputs=[st_messages, landing_page, main_app, st_uploaded_pdf_path, pdf_viewer, build_log]
    )

# 실행
gr.close_all()
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
