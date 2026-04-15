AutoRAG · 汽车售后智能问答系统

基于 RAG 架构的汽车使用手册问答系统,围绕多车型物理隔离与评测驱动迭代两个核心工程主题构建。

项目周期:2025.12 - 至今(持续迭代)
当前阶段:评测体系完成首轮 baseline,识别优化方向中

一、项目背景
汽车售后场景下,车主常需翻阅厚重的使用手册定位某个功能或故障的解释。本项目以问界 M8 / 问界 M9纯电版 / 尊界 S800 增程版三款车型的官方手册为知识源,提供自然语言问答能力。
为什么从扣子/Dify 切换到自研:
项目最初尝试使用扣子(Coze)和 Dify 搭建,但在以下场景上遇到瓶颈:

扣子知识库对长篇结构化文档(几百页 PDF)的解析与召回效果不稳定
多车型场景下无法做精细的索引隔离,容易出现跨车型答案污染
缺乏对检索/重排/生成各层的细粒度控制

因此切换为 Python + LangChain + FAISS 自研方案,并在第二阶段引入 Ragas 评测框架做工程化迭代。

二、核心亮点
1. 多车型物理隔离索引架构
三个车型的知识库独立建库、独立检索,从根本上避免跨车型知识污染。例如用户问"我的 M9 该换机油了吗"——M9 是纯电车,系统能识别这是该拒答的陷阱问题,而不是从 S800 增程版的手册里捞数据瞎编。
2. 完整的 Ragas 评测体系 + 自定义指标
不止跑通 Ragas 原生四指标(Faithfulness / Answer Relevancy / Context Precision / Context Recall),还自研了 Refusal Accuracy(拒答准确率) 指标,专门评测系统对跨车型陷阱题的拒答能力。
3. 22 条精心设计的多维度评测集
覆盖 7 种 hop_type:单跳问答、跨车型陷阱、反向陷阱、2-hop 多跳、能力边界题、错别字鲁棒性、模糊指代题。其中 10 条是人工设计的真实用户场景题,而非全靠 Ragas TestsetGenerator 自动生成。
4. 评测-迭代闭环工具链
沉淀了三个工具脚本:clean_testset.py / merge_testset.py / run_all.py,可复用至其他车型或其他知识源场景。

三、系统架构
┌──────────────────────────────────────────────────────────┐
│                  生产链路 (routes.py)                     │
│                                                           │
│   用户 query → FastAPI → 向量检索(FAISS)                 │
│                            ↓                              │
│                       Reranker(BGE-Reranker-base)        │
│                            ↓                              │
│                       GLM-4 流式生成                      │
│                            ↓                              │
│                       SSE 返回给用户                      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                评测链路 (evals/run_all.py)                │
│                                                           │
│   testset → 向量检索 → GLM-4 生成 → Ragas 4 指标         │
│                                  ↓                        │
│                           Refusal Accuracy               │
│                                  ↓                        │
│                          题级 CSV + 汇总 stdout          │
└──────────────────────────────────────────────────────────┘
架构注解:目前生产链路接了 Reranker,评测链路尚未接入(属于已知遗留项,见第六节"已知问题")。

四、评测结果(首轮 Baseline)
评测对象:m9ev 车型,22 条混合评测集
评测时间:2026-04-14
评测方式:裸向量检索(top_k=5),未接 Reranker,GLM-4 生成,DeepSeek 作为 Ragas judge
指标数值说明Faithfulness0.767答案与检索 context 的一致性Answer Relevancy0.578答案与问题的相关性Context Precision0.494检索 chunk 的相关比例Context Recall0.589正确信息的召回率Refusal Accuracy4/4 (100%)跨车型陷阱题全部正确拒答
数据诚实说明

这是裸向量检索的 baseline,生产链路实际带 Reranker,数字会更高
Context Precision 0.494 偏低,反映 chunk 切分还有优化空间(详见"已知问题")
Refusal Accuracy 100% 是基于关键词匹配判定,后续可升级到 LLM-as-judge 复测

题型分布
Hop Type题数来源ragas_single_hop12Ragas 自动生成 + 人工筛trap_cross_model4人工设计(跨车型陷阱)trap_reverse1人工设计(反向陷阱)multihop_22人工设计(2-hop 多跳)boundary_ambiguous1人工设计(模糊指代)capability_boundary1人工设计(辅助驾驶能力边界)boundary_typo1人工设计(错别字鲁棒性)

五、技术栈
层技术接口层FastAPI(SSE 流式)向量检索FAISSEmbeddingBAAI/bge-small-zhRerankerBAAI/bge-reranker-baseLLM智谱 GLM-4Ragas JudgeDeepSeek评测框架Ragas 0.2.x部署Docker演示界面Streamlit工具集成MCP Server

六、已知问题与下一步迭代
高优先级

Evaluation-Serving Skew:生产链路(routes.py)接了 Reranker,评测链路(run_all.py 调用的 search_chunks)未接。下一轮先做对齐,产出"生产代表性"的 v2 baseline。
Chunk 策略 v2:首版 baseline 暴露 chunk 切分问题(切到目录页、切断完整段落)。计划:按章节优先切、保护警告框/表格完整性、过滤目录索引页。
多跳题召回偏弱:Context Recall 在多跳题上明显偏低。计划接入 query 改写 + 多轮检索。

中优先级

Refusal Accuracy 升级到 LLM-as-judge,验证关键词匹配口径下的 100% 是否稳健。
补 m8 / s800evr 的 testset,做完整三车型评测对比。
接入对话记忆,处理多轮上下文(当前是单轮 RAG)。

低优先级 / 探索

Hybrid Search(BM25 + 向量)实验。
探索 Re-rank 后再做一次 LLM 评分过滤,削减低质量 context。


七、快速开始
方式一:Docker(推荐)
bashdocker build -t autorag .
docker run -p 8000:8000 --env-file .env autorag
方式二:本地
bashpip install -r requirements.txt
uvicorn app.main:app --reload
环境变量
项目根目录创建 .env:
ZHIPU_API_KEY=你的智谱AI密钥
DEEPSEEK_API_KEY=你的DeepSeek密钥(评测用)
跑评测
bashpython -m evals.run_all --model m9ev --testset evals/testset_m9ev_final.json

八、项目结构
autorag-rag/
├── app/                      # 生产服务
│   ├── main.py
│   ├── routes.py             # FastAPI 路由(带 Reranker)
│   └── services.py           # 检索 + 生成核心逻辑
├── evals/                    # 评测体系
│   ├── run_all.py            # 评测主入口
│   ├── clean_testset.py      # 清洗 Ragas 自动生成的 testset
│   ├── merge_testset.py      # 合并人工题与自动题
│   ├── gen_testset.py        # 调用 Ragas TestsetGenerator
│   ├── judge_config.py       # Ragas 0.2 兼容 monkey patch
│   ├── testset_m9ev_30.json  # Ragas 原始生成
│   ├── testset_m9ev_clean.json    # 清洗后(12 条)
│   ├── testset_m9ev_manual_10.json # 人工补充(10 条)
│   └── testset_m9ev_final.json     # 最终评测集(22 条)
├── mcp_server.py             # MCP 工具服务
├── streamlit_app.py          # Streamlit 演示界面
├── Dockerfile
└── requirements.txt

九、项目演进时间线
阶段时间关键动作构思2025.12培训期间识别需求,扣子/Dify 探索落地 v12026.03Python 自研 RAG,父子 chunk 策略 + Reranker重构 v22026.04切换为 Recursive Splitter,chunks 24337 → 779评测体系2026.04Ragas 框架 + 自定义 Refusal Accuracy + 22 条评测集第一轮 baseline2026.04.14跑通 baseline,识别 3 大优化方向下一轮 (TODO)2026.04+Evaluation-Serving 对齐 + chunk v2 + query 改写

作者
helenskuna251-ctrl · 2026 届应届生
求职方向:AI 应用工程师 / RAG 工程师 / LLM 应用开发(汽车行业背景优先,坐标杭州)
背景:主修汽车服务工程,2025.12 起自学转向 AI 应用开发,4 个月内独立完成 3 个项目(本项目 + LangGraph 多 Agent 客服系统 + ReAct 求职助手)。

这个项目是我从 0 到 1 的 RAG 工程实践,也是面试用的核心作品。如果您是招聘方,欢迎随时联系交流。