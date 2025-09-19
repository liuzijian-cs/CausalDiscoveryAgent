# src/llm/agents/knowledge_info.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-18 21:00
# 引用: https://github.com/Lancelot39/Causal-Copilot
# 描述: 知识信息提取 (Knowledge Information Extraction)

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


from ..LLMClient import LLMClient
from src.core.state import GlobalState

# Logger:
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# 任务1: 提取数据列信息 (Extract Data Column Information)
def get_full_knowledge(data_name, data_cols):
        client = LLMClient()
        prompt = ("I will conduct causal discovery on the Dataset %s containing the following Columns: \n\n"
                  "%s\n\nPlease provide comprehensive domain knowledge about this data. If variable names are meaningful, analyze in detail. If they're just symbols (like x1, y1), respond with 'No Knowledge'.\n\n"
                  "Please cover these aspects with clear structure:\n\n"
                  "1. VARIABLE DESCRIPTIONS: Detailed explanation of each variable, its meaning, measurement units, and typical ranges\n\n"
                  "2. CAUSAL RELATIONSHIPS: Potential direct and indirect causal connections between variables based on domain expertise\n\n"
                  "3. RELATIONSHIP NATURE: Are relationships primarily linear or nonlinear? Explain with examples\n\n"
                  "4. DATA DISTRIBUTION: Typical distributions of key variables (e.g., Gaussian, heavy-tailed, multimodal)\n\n"
                  "5. CONFOUNDERS: Potential unmeasured variables that might confound relationships\n\n"
                  "6. TEMPORAL ASPECTS: Time-dependencies, lags, or sequential relationships if relevant\n\n"
                  "7. HETEROGENEITY: Subgroups or contexts where relationships might differ\n\n"
                  "8. GRAPH DENSITY: Are causal relationships likely sparse (few connections) or dense (many connections)?\n\n"
                  "9. DOMAIN-SPECIFIC CONSTRAINTS: Physical laws, logical impossibilities, or theoretical frameworks that constrain possible causal relationships\n\n"
                  "10. RELEVANT LITERATURE: Key studies, papers, or established findings in this domain\n\n"
                  "11. DATA QUALITY ISSUES: Typical missing data patterns, measurement errors, or biases in this domain\n\n"
                  "12. INTERACTION EFFECTS: Complex variable interactions that might exist (multiplicative, threshold effects)\n\n"
                  "13. FEEDBACK LOOPS: Potential cyclic causal relationships that might exist\n\n"
                  "14. INSTRUMENTAL VARIABLES: Variables that might serve as valid instruments for causal identification\n\n"
                  "15. INTERVENTION HISTORY: Whether any variables reflect experimental interventions or policy changes\n\n"
                  "FOR TIME-SERIES DATA (if applicable):\n\n"
                  "16. STATIONARITY: Whether variables are expected to be stationary or have trends/seasonality\n\n"
                  "17. LAG STRUCTURE: Expected time lags between causes and effects in this domain\n\n"
                  "18. REGIME CHANGES: Known historical points where causal mechanisms might have changed\n\n"
                  "19. CONTEMPORANEOUS EFFECTS: Which variables might have instantaneous causal effects\n\n"
                  "20. PERIODICITY: Cyclical patterns or periodicities in the data generating process\n\n"
                  "Please organize your response by these numbered sections, with clear headings and concise, informative content in each section."
                  ) % (data_name, data_cols)
        
        response = client.chat_completion(
            prompt=prompt,
            system_prompt="You are an expert in the causal discovery field and helpful assistant.",
            json_response=False
        )
        return response

# 定义第二个LLM调用任务
def get_user_knowledge(data_name, data_cols):
    client = LLMClient()
    prompt = ("I will conduct causal discovery on the Dataset %s containing the following Columns: \n\n"
                "%s\n\nPlease provide comprehensive domain knowledge about this data. If variable names are meaningful, analyze in detail. If they're just symbols (like x1, y1), respond with 'No Knowledge'.\n\n"
                "Please cover these aspects with clear structure:\n\n"
                "1. VARIABLE DESCRIPTIONS: Detailed explanation of each variable, its meaning, measurement units, and typical ranges\n\n"
                "2. CAUSAL RELATIONSHIPS: Potential direct and indirect causal connections between variables based on domain expertise\n\n"
                "3. RELATIONSHIP NATURE: Are relationships primarily linear or nonlinear? Explain with examples\n\n"
                ) % (data_name, data_cols)
    
    response = client.chat_completion(
        prompt=prompt,
        system_prompt="You are a domain expert specializing in causal inference across multiple fields. Analyze dataset variables to extract comprehensive domain knowledge that would help with causal discovery.",
        json_response=False
    )
    return response

def knowledge_info(state: GlobalState):
    """提取知识信息 (Extract Knowledge Information)"""

    # 共享数据信息：
    data = state.user_data.processed_data
    data_name = state.user_data.data_name
    data_cols = data.columns.tolist() if data is not None else []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交任务
        future_full = executor.submit(get_full_knowledge, data_name, data_cols)
        future_user = executor.submit(get_user_knowledge, data_name, data_cols)

        # 收集结果（使用字典保存对应关系）
        futures = {
            future_full: 'full_knowledge',
            future_user: 'user_knowledge'
        }

        results = {}

        # 等待所有任务完成并获取结果
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                results[task_name] = result
                log.info(f"Task {task_name} completed successfully")
            except Exception as exc:
                log.error(f"Task {task_name} generated an exception: {exc}")
                log.error(traceback.format_exc())
                # 根据需求决定是否抛出异常或使用默认值
                raise exc
    
    # 设置全局状态
    state.user_data.knowledge_docs = results.get('full_knowledge')
    state.user_data.knowledge_docs_for_user = results.get('user_knowledge')








    