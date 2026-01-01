import streamlit as st
import json
import re
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 页面配置
st.set_page_config(
    page_title="个人知识库助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 标题
st.title("🤖 个人知识库助手 (My Knowledge Bot)")

# 左侧边栏
with st.sidebar:
    st.header("📄 文档上传")
    
    # 文件上传器
    uploaded_file = st.file_uploader(
        "上传 PDF 文档",
        type=["pdf"],
        help="支持上传 PDF 格式的文档"
    )
    
    # 提示文字
    st.caption("请上传你的文档，我会基于它回答问题。")
    
    # 如果上传了文件，显示文件信息
    if uploaded_file is not None:
        st.success(f"✅ 已上传: {uploaded_file.name}")
        st.info(f"文件大小: {uploaded_file.size / 1024:.2f} KB")

# 主区域
st.divider()

# 从 Streamlit Secrets 读取 API Key
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except KeyError:
    api_key = None
    st.error("⚠️ 管理员未配置密钥")

# PDF 解析和向量化逻辑
if uploaded_file is not None:
    try:
        with st.spinner("正在分析文档..."):
            # 读取 PDF 文件
            pdf_reader = PdfReader(uploaded_file)
            
            # 提取所有文本内容
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # 文本切片
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,  # <--- 加上这个！
                separators=["\n\n", "\n", "。", "！", "？"] # 尽量在句号处切分
            )
            chunks = text_splitter.split_text(text)
            
            # 向量化
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 创建向量索引
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            # 保存到 session_state，防止刷新丢失
            st.session_state.vectorstore = vectorstore
            
            # 显示成功提示
            st.success(f"✅ 成功建立索引！文档已切分为 {len(chunks)} 个片段。")
    
    except Exception as e:
        st.error(f"❌ 处理文档时出错: {str(e)}")

# 结构化数据提取功能
if "vectorstore" in st.session_state and api_key:
    st.divider()
    st.subheader("📊 结构化数据提取")
    
    if st.button("🔍 一键提取关键事件表", type="primary"):
        try:
            # 初始化 OpenAI 客户端
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            # 使用 RAG 检索相关文档片段（检索更多片段以获取完整信息）
            vectorstore = st.session_state.vectorstore
            # 检索更多片段以获取完整的事件信息
            relevant_chunks = vectorstore.similarity_search("关键事件 风险 应对措施", k=10)
            
            # 构建 Context
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # 构建结构化提取的 System Prompt
            system_prompt = """你是一个专业的数据提取助手。你的任务是从文档中提取关键事件信息，并严格按照指定的 JSON 格式输出。

要求：
1. 仔细分析文档内容，识别所有关键事件
2. 为每个事件评估风险等级（高/中/低）
3. 提取核心应对措施（不超过20字）
4. 记录事件所在的页码（如果文档中有页码信息）
5. **重要**：为每个事件评估置信度分数（1-10分），并引用原文依据
   - 如果能在文档中找到确切的原文依据，置信度应 >= 5
   - 如果找不到确切原文或信息不明确，置信度必须 < 5
   - evidence_quote 必须是从文档中直接引用的原话，不能是总结或改写

输出格式要求：
- 必须输出纯 JSON 格式，不要包含任何 Markdown 代码块标记（如 ```json）
- 不要输出任何解释性文字
- 严格按照以下 JSON Schema 输出：

{
  "events": [
    {
      "event_name": "事件名称",
      "risk_level": "高/中/低",
      "key_action": "核心应对措施(不超过20字)",
      "page_ref": 页码数字或null,
      "confidence_score": 1-10的整数,
      "evidence_quote": "从文档中直接引用的原文依据"
    }
  ]
}

如果文档中没有事件信息，返回空数组：{"events": []}"""

            user_prompt = f"请从以下文档内容中提取关键事件信息：\n\n{context}"
            
            with st.spinner("🔍 正在提取关键事件..."):
                # 调用 API，最多重试3次
                max_retries = 3
                json_data = None
                
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.1  # 降低温度以获得更稳定的输出
                        )
                        
                        # 获取响应内容
                        raw_response = response.choices[0].message.content.strip()
                        
                        # 清理可能的 Markdown 代码块标记
                        raw_response = re.sub(r'```json\s*', '', raw_response)
                        raw_response = re.sub(r'```\s*', '', raw_response)
                        raw_response = raw_response.strip()
                        
                        # 尝试解析 JSON
                        json_data = json.loads(raw_response)
                        break  # 成功解析，退出重试循环
                        
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            st.warning(f"⚠️ JSON 解析失败，正在重试... (尝试 {attempt + 1}/{max_retries})")
                            continue
                        else:
                            st.error(f"❌ JSON 解析失败：{str(e)}")
                            st.code(raw_response, language="text")
                            raise
                    except Exception as e:
                        if attempt < max_retries - 1:
                            st.warning(f"⚠️ 提取失败，正在重试... (尝试 {attempt + 1}/{max_retries})")
                            continue
                        else:
                            raise
            
            # 显示提取结果
            if json_data and "events" in json_data and len(json_data["events"]) > 0:
                st.success(f"✅ 成功提取 {len(json_data['events'])} 个关键事件！")
                
                # 转换为 DataFrame 格式
                events_list = json_data["events"]
                df = pd.DataFrame(events_list)
                
                # 确保 confidence_score 存在，如果不存在则设为默认值
                if "confidence_score" not in df.columns:
                    df["confidence_score"] = 5  # 默认值
                
                # 确保 evidence_quote 存在
                if "evidence_quote" not in df.columns:
                    df["evidence_quote"] = "未提供"
                
                # 处理 confidence_score 为数值类型
                df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(5)
                
                # 重新排列列的顺序，使其更易读
                column_order = ["event_name", "risk_level", "key_action", "confidence_score", "evidence_quote"]
                if "page_ref" in df.columns:
                    column_order.insert(3, "page_ref")
                
                # 只保留存在的列
                available_columns = [col for col in column_order if col in df.columns]
                df = df[available_columns]
                
                # 重命名列名为中文
                column_mapping = {
                    "event_name": "事件名称",
                    "risk_level": "风险等级",
                    "key_action": "核心应对措施",
                    "page_ref": "页码",
                    "confidence_score": "置信度",
                    "evidence_quote": "原文依据"
                }
                df = df.rename(columns=column_mapping)
                
                # 配置列显示格式
                column_config = {}
                
                # 配置置信度列为进度条
                if "置信度" in df.columns:
                    column_config["置信度"] = st.column_config.ProgressColumn(
                        "置信度",
                        help="模型对提取准确性的信心评分（1-10分）",
                        min_value=1,
                        max_value=10,
                        format="%d 分"
                    )
                
                # 配置原文依据列，使其可以展开查看
                if "原文依据" in df.columns:
                    column_config["原文依据"] = st.column_config.TextColumn(
                        "原文依据",
                        help="从文档中直接引用的原文",
                        width="large"
                    )
                
                # 在事件名称列中添加警告图标（如果置信度低）
                if "置信度" in df.columns and "事件名称" in df.columns:
                    # 为低置信度的事件添加警告图标
                    def add_warning_icon(row):
                        if pd.notna(row["置信度"]) and row["置信度"] < 7:
                            return f"⚠️ {row['事件名称']}"
                        return row["事件名称"]
                    
                    df["事件名称"] = df.apply(add_warning_icon, axis=1)
                
                # 显示表格
                st.dataframe(
                    df,
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 显示低置信度警告
                if "置信度" in df.columns:
                    low_confidence_df = df[df["置信度"] < 7]
                    low_confidence_count = len(low_confidence_df)
                    if low_confidence_count > 0:
                        st.warning(f"⚠️ 有 {low_confidence_count} 个事件的置信度低于 7 分，已用 ⚠️ 标记，请谨慎参考。")
                        # 显示低置信度事件的详细信息
                        with st.expander(f"🔍 查看低置信度事件详情 ({low_confidence_count} 个)"):
                            for idx, row in low_confidence_df.iterrows():
                                st.markdown(f"**{row.get('事件名称', '未知事件').replace('⚠️ ', '')}**")
                                st.caption(f"置信度: {row['置信度']:.0f}/10")
                                if "原文依据" in row and pd.notna(row["原文依据"]):
                                    st.info(f"原文依据: {row['原文依据']}")
                                st.divider()
                
                # 显示原始 JSON（可选，用于调试）
                with st.expander("📋 查看原始 JSON 数据"):
                    st.json(json_data)
            else:
                st.info("ℹ️ 未在文档中发现关键事件信息。")
                
        except Exception as e:
            st.error(f"❌ 提取过程中出错: {str(e)}")

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 聊天输入框
user_question = st.chat_input("向文档提问...")

# RAG 问答逻辑（向量检索模式）
if user_question:
    # 检查 API Key
    if not api_key:
        st.warning("⚠️ 管理员未配置密钥")
    # 检查向量库是否存在
    elif "vectorstore" not in st.session_state:
        st.warning("⚠️ 请先上传并解析 PDF 文档")
    else:
        try:
            # 保存用户问题到历史记录
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # 显示用户问题
            with st.chat_message("user"):
                st.write(user_question)
            
            # 向量检索：找出最相关的 3 个片段
            vectorstore = st.session_state.vectorstore
            relevant_chunks = vectorstore.similarity_search(user_question, k=3)
            
            # 构建 Context：拼接检索到的片段
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # 构建 Prompt
            prompt = f"基于以下参考片段回答问题：\n\n{context}\n\n问题：{user_question}"
            
            # 初始化 OpenAI 客户端（适配 DeepSeek）
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            # 调用 DeepSeek API
            with st.chat_message("assistant"):
                with st.spinner("🤔 正在思考中..."):
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                
                # 获取 AI 回答
                ai_answer = response.choices[0].message.content
                
                # 显示 AI 回答
                st.write(ai_answer)
            
            # 显示来源片段（在 AI 回答下方）
            with st.expander("🔍 查看 AI 参考的文档片段 (Source Context)"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**片段 {i}:**")
                    st.info(chunk.page_content)
                    if i < len(relevant_chunks):
                        st.markdown("---")
            
            # 保存 AI 回答到历史记录
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            
        except Exception as e:
            st.error(f"❌ 调用 API 时出错: {str(e)}")

