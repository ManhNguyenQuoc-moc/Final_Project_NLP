import streamlit as st
import tensorflow as tf
from tokenizers import Tokenizer
from model import transformer  # Đảm bảo rằng bạn có model transformer.py
import numpy as np
import os

# Cấu hình
TOKENIZER_PATH = "bpe_tokenizer.json"
MODEL_CHECKPOINT = "checkpoints/checkpoint-50.h5"
MAX_LEN = 128
BEAM_WIDTH = 3
NUM_LAYERS = 4
D_MODEL = 128
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        return tokenizer
    except Exception as e:
        st.error(f"Không thể tải tokenizer: {e}")
        return None

# Load model
@st.cache_resource
def load_model(vocab_size):
    model = transformer(
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )
    if not os.path.exists(MODEL_CHECKPOINT):
        st.error(f"Không tìm thấy mô hình checkpoint tại {MODEL_CHECKPOINT}.")
        return None
    model.load_weights(MODEL_CHECKPOINT)
    return model

# Beam search decoder
def beam_search_decoder(input_text, model, tokenizer, max_length=MAX_LEN, beam_width=BEAM_WIDTH):
    inputs = tokenizer.encode(input_text)
    input_ids = tf.constant([inputs.ids], dtype=tf.int32)

    start_token = tokenizer.token_to_id("<bos>")
    end_token = tokenizer.token_to_id("<eos>")
    beams = [(tf.constant([[start_token]], dtype=tf.int32), 0.0)]  # (sequence, score)

    for _ in range(max_length):
        new_beams = []
        for sequence, score in beams:
            if sequence[0, -1] == end_token:
                new_beams.append((sequence, score + 1.0))  # Ưu tiên chuỗi đã kết thúc
                continue

            predictions = model([input_ids, sequence], training=False)
            logits = predictions[:, -1, :]  # Logits của token cuối
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            top_k_probs, top_k_indices = tf.nn.top_k(log_probs, k=beam_width)
            top_k_probs = top_k_probs[0].numpy()
            top_k_indices = top_k_indices[0].numpy()

            for prob, idx in zip(top_k_probs, top_k_indices):
                new_sequence = tf.concat([sequence, tf.constant([[idx]], dtype=tf.int32)], axis=-1)
                new_score = score + prob
                new_beams.append((new_sequence, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if all(seq[0, -1] == end_token for seq, _ in beams):
            break

    best_sequence, _ = beams[0]
    output_ids = best_sequence[0, 1:].numpy()  # Bỏ <bos>
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    if not output_text.strip():
        output_text = "Xin lỗi, mình chưa hiểu rõ câu hỏi. Cậu có thể nói rõ hơn không?"

    return output_text

# Streamlit interface
def main():
    st.set_page_config(page_title="Chatbot Tư vấn Tình cảm", layout="centered", page_icon="💌")
    st.title("💌 Chatbot Tình Cảm")
    st.markdown(
        """
        Chào bạn! Mình là chatbot chuyên tư vấn về tình cảm.  
        Hãy chia sẻ câu chuyện hoặc câu hỏi của bạn, mình sẽ trả lời thật chân thành! 💬  
        Ví dụ: "Mình thích một người nhưng không biết làm sao để bày tỏ..."
        """
    )

    tokenizer = load_tokenizer()
    if tokenizer is None:
        st.error("Không thể tải tokenizer. Hãy kiểm tra lại file `bpe_tokenizer.json`. ")
        return

    model = load_model(tokenizer.get_vocab_size())
    if model is None:
        st.error("Không thể tải mô hình. Hãy kiểm tra lại file checkpoint.")
        return

    # Khởi tạo lịch sử hội thoại nếu chưa có
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input form
    with st.form(key="chat_form"):
        user_input = st.text_area(
            "Câu hỏi hoặc tâm sự của bạn:",
            height=150,
            placeholder="Nhập câu hỏi hoặc câu chuyện của bạn tại đây..."
        )
        submit_button = st.form_submit_button("Gửi")

    if submit_button:
        if user_input.strip():
            with st.spinner("Đang suy nghĩ..."):
                response = beam_search_decoder(user_input, model, tokenizer)
                # Thêm vào lịch sử trò chuyện
                st.session_state.chat_history.append({"user": user_input, "bot": response})
                st.markdown("**🤖 Chatbot trả lời:**")
                st.success(response)
        else:
            st.warning("Vui lòng nhập nội dung trước khi gửi!")

    # Hiển thị lịch sử trò chuyện
    if st.session_state.chat_history:
        st.markdown("### 🗨️ Lịch sử trò chuyện")
        for chat in st.session_state.chat_history:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Bạn:** {chat['user']}")
            with st.chat_message("assistant", avatar="💬"):
                st.markdown(f"**Chatbot:** {chat['bot']}")

if __name__ == "__main__":
    main()
