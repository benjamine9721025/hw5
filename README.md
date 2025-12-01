# hw5

https://7sx9drgir7sksbqcezbbcq.streamlit.app/


# 🀄 中文 AI 文本偵測器（Chinese AI Text Detector）

本專案提供一個能夠判斷 **中文文本是「AI 生成」或「人類撰寫」** 的線上工具，採用 HuggingFace 上的  
**`yuchuantian/AIGC_detector_zhv2`** 中文 AI 文本偵測模型（BERT Base 微調，用於人類 / AI 二分類），並以 **Streamlit** 部署成互動式網頁介面。

## 🚀 功能特色

- 支援 **中文文本** 偵測（非英文模型硬套）
- 基於 BERT 的 AI 生成文本分類模型
- 自動輸出兩類別：
  - **人類撰寫**
  - **AI 生成**
- 同時顯示兩者的 **機率分佈**
- 可本機執行，也可部署至 Streamlit Cloud
- 介面操作簡單、適合教學／研究用途

## 📂 專案結構

- hw5/
- ├── app.py # Streamlit 主程式
- ├── requirements.txt # 依賴套件列表
- └── README.md # 說明文件（本檔案）

## 📘 app.py 功能說明

-使用 BertTokenizer + BertForSequenceClassification
-模型輸出兩個 label：
-Label 0 → 人類撰寫（Human）
-Label 1 → AI 生成（AI-Generated）
-使用 softmax 計算機率
-顯示結果（類別 + 機率）
-顯示機率條狀圖
-支援多行中文輸入


## 📦 依賴模型

-本專案使用：

-🔹 yuchuantian/AIGC_detector_zhv2
-專為中文 AI 文本偵測訓練的 BERT 模型
-來源：HuggingFace
-用途：判斷文本為 AI 生成 或 人類撰寫


##⚠️ 限制與使用說明

-模型輸出為機率，不是百分之百準確
-段落越短，偵測越不穩定
-不應將結果用作學術違規或法律判定的唯一依據
-不保證能準確辨識所有 LLM（尤其是微調後的模型）
