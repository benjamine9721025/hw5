import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# ----------------------------------------------------
# 0. 模型設定：中文 AI 文本偵測器
# ----------------------------------------------------
# 來自論文《Multiscale Positive-Unlabeled Detection of AI-Generated Texts》
# 官方 Space 指出 zh v2 是效果很好的中文 AI 偵測器之一:contentReference[oaicite:1]{index=1}
MODEL_NAME_ZH = "yuchuantian/AIGC_detector_zhv2"


# ----------------------------------------------------
# 1. 中文文本前處理（簡單版）
# ----------------------------------------------------
def clean_zh_text(text: str) -> str:
    # 去掉前後空白／換行，保留中文內容
    return text.strip()


# ----------------------------------------------------
# 2. 載入中文偵測模型（用 Streamlit cache，避免重覆下載）
# ----------------------------------------------------
@st.cache_resource
def load_zh_detector():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_ZH)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME_ZH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ----------------------------------------------------
# 3. 預測函式：輸入一段中文，輸出「人類」或「AI」以及機率
# ----------------------------------------------------
def predict_zh_ai_score(text: str):
    tokenizer, model, device = load_zh_detector()
    cleaned = clean_zh_text(text)
    if not cleaned:
        return None

    # tokenizer 會自動做中文斷詞 / 編碼
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, 2)

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()

    # 官方程式碼標註的順序是：0 = Human, 1 = AI:contentReference[oaicite:2]{index=2}
    id2label = ["人類撰寫", "AI 生成"]
    score_dict = {
        id2label[0]: float(probs[0]),
        id2label[1]: float(probs[1]),
    }

    # 取最大值當主預測
    if score_dict[id2label[0]] >= score_dict[id2label[1]]:
        best_label = id2label[0]
        best_score = score_dict[id2label[0]]
    else:
        best_label = id2label[1]
        best_score = score_dict[id2label[1]]

    return best_label, best_score, score_dict


# ----------------------------------------------------
# 4. Streamlit 介面
# ----------------------------------------------------
st.set_page_config(
    page_title="中文 AI 文本偵測器",
    page_icon="🀄",
    layout="wide",
)

st.title("🀄 中文 AI 文本偵測器")
st.caption(
    "模型：`yuchuantian/AIGC_detector_zhv2`（針對中文訓練的人類 vs AI 文本偵測模型）"
)

with st.expander("📌 使用說明 / 限制", expanded=True):
    st.markdown(
        """
- 本工具用來估計一段**中文文本**比較像「人類撰寫」或「AI 生成」。
- 分數是模型的機率估計，**不是 100% 準確的真實標記**。
- 偵測結果**不適合當作學術違規或嚴重指控的唯一證據**，請一定搭配人工判斷。
- 長度太短（例如只有一兩句）時，任何 AI 偵測器都會比較不穩定。
        """
    )

default_zh_text = """人工智慧技術的發展，讓自動生成文本變得越來越普遍。
這段文字是用來測試中文 AI 文本偵測器的示例內容，並不代表真實立場。
"""

text = st.text_area(
    "請貼上要檢測的中文文本：",
    value=default_zh_text,
    height=250,
    placeholder="建議貼上至少幾句話，篇幅長一點偵測會比較穩定。",
)

col1, col2 = st.columns([1, 2])

with col1:
    analyze_btn = st.button("🔍 開始偵測", use_container_width=True)

if analyze_btn:
    if not text.strip():
        st.warning("請先輸入要檢測的中文文本。")
    else:
        with st.spinner("模型推論中，請稍候..."):
            result = predict_zh_ai_score(text)

        if result is None:
            st.error("文本為空，無法偵測。")
        else:
            best_label, best_score, score_dict = result

            # ----- 左側：主結果 -----
            with col1:
                st.subheader("🔎 判斷結果")
                st.markdown(f"**預測類別**：`{best_label}`")
                st.markdown(f"**置信度（機率）**：`{best_score:.4f}`")

            # ----- 右側：兩類別分數條狀圖 -----
            with col2:
                st.subheader("📊 分數分佈")
                df = pd.DataFrame(
                    {
                        "label": list(score_dict.keys()),
                        "score": list(score_dict.values()),
                    }
                ).set_index("label")
                st.bar_chart(df)

            st.divider()
            st.subheader("📝 送出文本（節錄）")
            st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
else:
    st.info("輸入中文文本後，按下「🔍 開始偵測」即可看到結果。")
