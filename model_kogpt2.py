from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_answer(user_input: str) -> str:
    prompt = (
        "당신은 정중하고 정확한 한국어 AI입니다.\n"
        "Q: 대한민국의 수도는 어디인가요?\n"
        "A: 대한민국의 수도는 서울입니다.\n"
        f"Q: {user_input}\n"
        "A:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else None
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("A:")[-1].strip()
    return answer


# CLI 테스트용 코드 (필요시 사용)
if __name__ == "__main__":
    print("🤖 KoGPT2 모델 로딩 완료")
    while True:
        prompt = input("💬 질문 입력 (종료: exit): ")
        if prompt.lower() == "exit":
            break
        response = generate_answer(prompt)
        print("🤖 답변:", response)
