import sys
sys.path.append("/data/npl/ICEK/News/NLP/my-lora-project")
from src.inference import generate_text

if __name__ == "__main__":
    prompt = """
        <|im_start|>system
        Bạn là một trợ lý AI hữu ích. Hãy trả lời người dùng một cách chính xác.
        <|im_end|>

        <|im_start|>user
        Hãy miêu tả thành phố Đà Lạt chi tiết.
        <|im_end|>

        <|im_start|>assistant
        """
    output = generate_text(prompt, max_new_tokens=150)
    print("===== Generated Output =====")
    print(output)
