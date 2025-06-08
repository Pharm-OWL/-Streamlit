
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="藥品儲位優化原型", layout="wide")
st.title("💊 藥品儲位優化系統（模擬互動原型）")
st.markdown("模擬展示關聯規則如何改善藥品儲位配置與取藥動線")

uploaded_file = st.file_uploader("請上傳處方 CSV 檔（欄位名稱：處方內容，藥品以逗號分隔）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("模擬處方資料.csv")

st.subheader("📋 處方資料預覽")
st.dataframe(df.head(10))

transactions = df["處方內容"].dropna().apply(lambda x: [i.strip() for i in x.split(",")]).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_tf = pd.DataFrame(te_ary, columns=te.columns_)

st.sidebar.header("🔍 關聯規則分析參數")
min_support = st.sidebar.slider("最小支持度 (support)", 0.01, 0.5, 0.1, 0.01)
min_confidence = st.sidebar.slider("最小信賴度 (confidence)", 0.1, 1.0, 0.5, 0.05)

frequent_itemsets = apriori(df_tf, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules = rules.sort_values(by="lift", ascending=False)

st.subheader("📈 關聯規則結果")
st.write("關聯規則數量：", len(rules))
if not rules.empty:
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
    st.subheader("🔥 藥品使用熱度")
    top_items = df_tf.sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_items.values, y=top_items.index, ax=ax, palette="Blues_r")
    ax.set_xlabel("出現次數")
    st.pyplot(fig)
else:
    st.warning("沒有找到符合條件的關聯規則，請嘗試降低門檻。")

st.subheader("🗺️ 儲位優化建議（模擬展示）")
if not rules.empty:
    top_pairs = rules.head(5)[["antecedents", "consequents"]]
    for idx, row in top_pairs.iterrows():
        a = ", ".join(list(row["antecedents"]))
        b = ", ".join(list(row["consequents"]))
        st.write(f"👉 建議將 **{a}** 與 **{b}** 擺放在相近儲位")
else:
    st.info("無可提供之儲位建議。")

st.caption("📌 本系統為展示用原型，資料與分析結果皆為模擬。")
