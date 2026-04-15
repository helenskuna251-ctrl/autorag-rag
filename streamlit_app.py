import streamlit as st
import requests

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoRAG · 汽车使用手册智能问答",
    page_icon="🚗",
    layout="centered"
)

API_BASE = "http://localhost:8000"  # 你的 FastAPI 地址，按需修改

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.title("🚗 AutoRAG · 汽车使用手册智能问答")
st.caption("上传车辆使用手册 PDF，即可开始智能问答")

# ── 侧边栏：车型管理 & 文件上传 ───────────────────────────────────────────────
with st.sidebar:
    st.header("📁 知识库管理")

    car_model = st.text_input(
        "车型名称",
        value="general",
        placeholder="例如：BMW_3系、Audi_A4",
        help="每个车型对应独立的知识库索引"
    )

    uploaded_file = st.file_uploader(
        "上传使用手册 PDF",
        type=["pdf"],
        help="支持 PDF 格式"
    )

    if st.button("📤 上传并建立索引", use_container_width=True):
        if not uploaded_file:
            st.warning("请先选择 PDF 文件")
        elif not car_model.strip():
            st.warning("请填写车型名称")
        else:
            with st.spinner(f"正在解析并建立 [{car_model}] 索引，请稍候..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    params = {"car_model": car_model}
                    resp = requests.post(f"{API_BASE}/upload", files=files, params=params, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ 索引建立成功！")
                        st.info(f"chunks:{data.get('chunks')} 个")
                        # 记录已上传车型
                        if car_model not in st.session_state.get("available_models", []):
                            if "available_models" not in st.session_state:
                                st.session_state.available_models = []
                            st.session_state.available_models.append(car_model)
                    else:
                        st.error(f"上传失败：{resp.text}")
                except Exception as e:
                    st.error(f"连接失败，请确认 FastAPI 服务已启动：{e}")

    st.divider()

    # 已有车型选择
    st.subheader("🔍 切换查询车型")
    available = st.session_state.get("available_models", [])
    if available:
        selected_model = st.selectbox("选择车型", options=available)
    else:
        selected_model = car_model
        st.caption("暂无已上传车型，使用当前填写的车型名称")

    st.divider()
    if st.button("🗑️ 清空对话记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 对话历史初始化 ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 展示历史消息 ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 输入框 ────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("请输入你的问题，例如：如何开启自动泊车？"):

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用流式接口
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_BASE}/query/stream",
                json={"query": prompt},
                params={"car_model": selected_model},
                stream=True,
                timeout=60
            ) as resp:
                if resp.status_code == 200:
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                else:
                    error_msg = f"请求失败（{resp.status_code}）：{resp.text}"
                    placeholder.error(error_msg)
                    full_response = error_msg

        except requests.exceptions.ConnectionError:
            full_response = "❌ 无法连接到后端服务，请确认 FastAPI 已启动（默认 http://localhost:8000）"
            placeholder.error(full_response)
        except Exception as e:
            full_response = f"❌ 出错了：{e}"
            placeholder.error(full_response)

    # 保存 assistant 回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})
