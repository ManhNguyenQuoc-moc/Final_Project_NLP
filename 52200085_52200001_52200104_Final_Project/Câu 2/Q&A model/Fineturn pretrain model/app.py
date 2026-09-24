import streamlit as st
import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

# ====== Kiểm tra phiên bản PyTorch ======
# ====== Cấu hình trang phải là lệnh Streamlit đầu tiên ======
st.set_page_config(page_title="Chatbot Tư vấn Tình cảm", layout="centered", page_icon="💌")

# ====== Hàm tiền xử lý ======
def normalize_text(text):
    """
    Chuẩn hóa Unicode và loại bỏ ký tự đặc biệt ngoài .,!?
    """
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[^\w\s\.,!\?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def denormalize_response(text):
    """
    Định dạng lại câu trả lời: thay '_' thành khoảng trắng,
    viết hoa đầu câu và xử lý dấu câu sát từ trước.
    """
    text = text.replace('_', ' ')
    text = re.sub(r"\s+([\.,!\?])", r"\1", text)
    text = re.sub(r"\s+", ' ', text)
    parts = re.split(r'([\.!\?])', text)
    formatted = []
    for i in range(0, len(parts)-1, 2):
        sentence = parts[i].strip()
        punct = parts[i+1]
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            formatted.append(sentence + punct)
    return ' '.join(formatted).strip()

# ====== Load mô hình và tokenizer ======
@st.cache_resource
def load_model_and_tokenizer(model_dir: str):
    """
    Load tokenizer và mô hình từ thư mục đã lưu pretrained.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục mô hình: {model_dir}")
    # AutoTokenizer.from_pretrained sẽ kiểm tra config.json
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model

# ====== Giao diện Streamlit ======
st.title("💌 Chatbot Tình Cảm")
st.markdown(
    """
    Chào bạn! Mình là chatbot chuyên tư vấn về tình cảm.  
    Hãy chia sẻ câu chuyện hoặc câu hỏi của bạn, mình sẽ trả lời thật chân thành! 💬  
    Ví dụ: "Mình thích một người nhưng không biết làm sao để bày tỏ..."
    """
)

# Khởi tạo lịch sử chat (nếu cần)
if 'history' not in st.session_state:
    st.session_state.history = []

# ====== Tải tokenizer và mô hình sau khi page config ======
MODEL_DIR = './checkpoint-14000'
try:
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
except Exception as e:
    st.error(f"Lỗi khi load model/tokenizer: {e}")
    st.stop()

# ====== Hàm sinh câu trả lời ======
def generate_response(question: str, max_length: int = 200):
    """
    Sinh câu trả lời từ câu hỏi, chỉ sử dụng prompt mới.
    """
    q = normalize_text(question)
    prompt = f"<|startoftext|><|question|> {q} <|response|>"
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        resp = raw.split('<|response|>')[-1].split('<|endoftext|>')[0].strip()
    except:
        resp = raw
    return denormalize_response(resp)

# ====== Form nhập liệu ======
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area(
        "💡 Bạn muốn chia sẻ điều gì hôm nay?",
        height=150,
        placeholder="Ví dụ: Mình đang có vấn đề về tình cảmcảm..."
    )
    send = st.form_submit_button("Gửi 💌")

if send:
    if not user_input.strip():
        st.warning("⛔ Vui lòng nhập câu hỏi hoặc chia sẻ.")
    else:
        with st.spinner("🤔 Đang suy nghĩ..."):
            answer = generate_response(user_input)
        st.markdown("**🤖 Chatbot trả lời:**")
        st.success(answer)
        st.session_state.history.append({'user': user_input, 'bot': answer})

# ====== Hiển thị lịch sử chat (nếu cần) ======
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🗨️ Lịch sử trò chuyện")
    for chat in st.session_state.history:
        st.markdown(f"**Bạn:** {chat['user']}")
        st.markdown(f"**Chatbot:** {chat['bot']}\n")
