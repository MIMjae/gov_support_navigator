# model.py
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/polyglot-ko-1.3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_answer(user_input: str) -> str:
    system_prompt = (
        "당신은 사용자의 질문에 대해 정확하고 정중한 말투로 답변하는 한국어 인공지능 비서입니다.\n"
        "모든 문장은 공식적인 어투(입니다/습니다)를 사용하십시오.\n"
        "맞춤법과 문법을 올바르게 사용하십시오.\n"
        "질문에 대해 간결하면서도 핵심 정보를 빠짐없이 전달하십시오.\n\n"
        "5문장 이내로 답변하며, 필요하지 않은 정보는 제외하십시오.\n\n"
        "답변이 끝나면, 온점 뒤의 특수문자는 제외하십시오.\n\n"
        f"질문: {user_input}\n답변:"
    )

    inputs = tokenizer(system_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.8,
        top_k=40
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace(system_prompt, "").strip()


# ✅ 테스트 CLI 실행
# if __name__ == "__main__":
#     print("🤖 모델 로딩 완료 (Polyglot-ko)")
#     while True:
#         prompt = input("💬 질문 입력 (종료: exit): ")
#         if prompt.lower() == "exit":
#             break
#         response = generate_answer(prompt)
#         print("🤖 답변:", response)
