import os
import json
import time
import random
from openai import OpenAI
from dotenv import load_dotenv

# --- 설정 ---
load_dotenv()
client = OpenAI()

RAW_TEXTS_DIR = "raw_texts"
OUTPUT_FINETUNE_FILE = "finetuning_dataset_300.jsonl"
TARGET_DATASET_SIZE = 300
MODEL_FOR_GENERATION = "gpt-4o-mini"
NUM_SENTENCES_PER_CHUNK = 5

# --- 함수 정의 ---
def load_and_combine_all_sentences(directory):
    """지정된 폴더의 모든 txt 파일에서 문장들을 읽어 하나의 리스트로 합칩니다."""
    all_sentences = []
    print(f"'{directory}' 폴더에서 텍스트 파일들을 읽어옵니다...")
    if not os.path.exists(directory):
        print(f"[오류] '{directory}' 폴더를 찾을 수 없습니다.")
        return None
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
                all_sentences.extend(sentences)
    return all_sentences

def generate_finetuning_pair(sentence_chunk):
    """
    주어진 문장 묶음의 핵심을 파악하여,
    스티브 잡스 스타일의 '마케팅 원칙' (output)과 그에 맞는 '지시' (instruction)를 생성합니다.
    """
    context = " ".join(sentence_chunk)
    
    system_prompt = (
        "You are an AI that embodies the spirit of Steve Jobs. Your task is to distill the core marketing or branding principle from a given text, "
        "and then rephrase it as a short, powerful, and insightful piece of advice in KOREAN. "
        "The tone should be direct, confident, and focused on the essence. Use simple words and strong metaphors. "
        "Crucially, the entire output must be in a formal, respectful Korean tone (존댓말). "
        "The final output must be a single JSON object with two keys: 'instruction' and 'output'."
    )
    
    user_prompt = f"""
    Here is an excerpt from a keynote speech:
    ---
    {context}
    ---
    Based on the text above, generate a JSON object with:
    1. "instruction": A Korean question that would elicit this principle. (e.g., "신제품의 핵심 가치를 어떻게 전달해야 할까요?")
    2. "output": The core principle, rewritten in Steve Jobs's iconic style in 2-3 concise Korean sentences. (Must use formal Korean speech - 존댓말)
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_FOR_GENERATION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        if "instruction" in result and "output" in result:
            return result
        else:
            return None
    except Exception as e:
        print(f"  - API 호출 오류: {e}")
        return None


# --- 실행 ---
all_sentences = load_and_combine_all_sentences(RAW_TEXTS_DIR)

if all_sentences and len(all_sentences) >= NUM_SENTENCES_PER_CHUNK:
    print(f"총 {len(all_sentences)}개의 문장을 기반으로 데이터셋 생성을 시작합니다...")
    
    generated_count = 0
    with open(OUTPUT_FINETUNE_FILE, "w", encoding="utf-8") as f_out:
        # TARGET_DATASET_SIZE 만큼만 반복
        while generated_count < TARGET_DATASET_SIZE:
            # 리스트에서 랜덤한 위치를 골라 문장 묶음을 추출
            start_index = random.randint(0, len(all_sentences) - NUM_SENTENCES_PER_CHUNK)
            chunk = all_sentences[start_index : start_index + NUM_SENTENCES_PER_CHUNK]
            
            print(f"[{generated_count + 1}/{TARGET_DATASET_SIZE}] 데이터 생성 중...")
            
            # LLM을 이용해 instruction과 output 생성
            pair = generate_finetuning_pair(chunk)
            
            if pair:
                f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                generated_count += 1
            
            # API 속도 제한 방지
            time.sleep(1)

    print(f"\n🎉 파인튜닝 데이터셋 생성 완료! '{OUTPUT_FINETUNE_FILE}' 파일을 확인하세요.")
else:
    print("처리할 문장이 부족합니다. `raw_texts` 폴더에 파일을 확인해주세요.")