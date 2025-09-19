# src/llm/agents/algorithm_selector.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-19 13:30
# 描述: 自动化代码选择器 (Automated Code Selector)

import logging
from ..LLMClient import LLMClient
from src.core.state import GlobalState

# Logger:
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SYSTEM_PROMPT = """
<role>
You are an experienced expert in causal discovery algorithms, familiar with the strengths, weaknesses, applicable scenarios, and implementation details of various causal discovery methods. Your task is to recommend the most suitable causal discovery algorithm based on the data characteristics and requirements provided by the user. Your reply format needs to strictly follow the JSON format specifed below.
</role>

<candidate_algorithms>
All candidate algorithms, their descriptions and tags:
[ALGO_CONTEXT]
</candidate_algorithms>


<dataset_characteristics>
For the dataset [TABLE_NAME] that have the following variables:
[COLUMNS]
And the following statistics:
[STATISTICS_DESC]
</dataset_characteristics>

<domain_knowledge>
[DOMAIN_KNOWLEDGE]
</domain_knowledge>

<cuda_availability>
[CUDA_WARNING]
</cuda_availability>

<causal_graph>
[ACCEPT_CPDAG]
</causal_graph>

<steps>
I need you to carefully analyze and select the most suitable causal discovery algorithms (up to [TOP_K]) through a comprehensive multi-step reasoning and decision process as follow:

1. **Data Characteristics Analysis**:
   - Sample size (n): Is it sufficient for statistical power? (small: <500, medium: 500-5000, large: >5000)
   - Variable count (p): How many variables need to be considered? Consider these thresholds:
     * Small scale (<25 variables): Most algorithms perform well
     * Medium scale (25-50 variables): Requires "High" or better scalability rating
     * Large scale (50-110 variables): Requires "Very High" or better scalability rating
     * Very large scale (>110 variables): Requires "Extreme" scalability rating
   - Variable types: Continuous, discrete, categorical, mixed? What proportion of each?
   - Potential confounders: Are there likely unmeasured confounding variables?
   - Graph density: Is the graph underlying this data/domain likely to be a dense graph or a sparse graph?

2.  **Resource Constraints**:
   - Computational resources: GPU availability, memory limitations, time constraints
   - Output format requirements: Is a DAG, CPDAG, or PAG preferred or required?

</steps>

<important>
FOCUS EXCLUSIVELY ON THE CURRENT DATASET CHARACTERISTICS. PRIORITIZE ALGORITHMIC DIVERSITY by selecting algorithms from different methodological families (e.g., score-based, constraint-based, continous-optimization-based...) when multiple algorithms are equally compatible with the requirements.

Your final response should include the complete reasoning process, for each algorithm, include justification, description, and selected algorithm in a JSON object, JSON format is as follows:

{
  "reasoning": "Detailed step-by-step reasoning process",
  "algorithms": [
    {
      "justification": "Comprehensive explanation connecting THIS dataset's specific characteristics to algorithm strengths and showing why this algorithm outperforms alternatives for this particular use case.",
      "description": "Concise description of the algorithm's approach and capabilities.", 
      "name": "Algorithm Name (Key name of candidates, not full name)",
    },
    ...
  ]
}
</important>


"""
# - Missing data: What percentage of values are missing? How are they distributed?

def parse_response(algo_candidates):
    return {algo['name']: {
        'description': algo['description'],
        'justification': algo['justification']
    } for algo in algo_candidates['algorithms']}

def algorithm_selector(state: GlobalState, top_k: int = 3):
    """根据数据特征和用户需求选择合适的算法 (Select appropriate algorithm based on data characteristics and user needs)"""
    client = LLMClient()
    
    # 读取算法描述文件 (Read algorithm description file)
    with open("src/causal_discovery/context/algorithms/description.txt", "r", encoding="utf-8") as f:
        algorithm_description = f.read()
    
    replacements = {
        "[TABLE_NAME]": state.user_data.data_name or "Unknown Dataset",
        "[COLUMNS]": ', '.join(state.user_data.processed_data.columns) if state.user_data.processed_data is not None else "Unknown Columns",
        "[STATISTICS_DESC]": state.statistics.description or "No statistical description available.",
        "[DOMAIN_KNOWLEDGE]": state.user_data.knowledge_docs or "No domain knowledge available.",
        "[ALGO_CONTEXT]": algorithm_description,
        "[CUDA_WARNING]": "GPU is available." if state.algorithm.gpu_available else "WARNING: GPU is NOT available. Only CPU-based algorithms can be selected.",
        "[TOP_K]": str(top_k),
        "[ACCEPT_CPDAG]": "The user accepts the output graph including undirected edges/undeterministic directions (CPDAG/PAG)" if state.user_data.accept_cpdag else "The user does not accept the output graph including undirected edges/undeterministic directions (CPDAG/PAG), so the output graph should be a DAG."
    }

    system_prompt = SYSTEM_PROMPT
    for placeholder, value in replacements.items():
        system_prompt = system_prompt.replace(placeholder, value)

    response = client.chat_completion(
        prompt=f"Please choose the algorithms that provide most reliable and accurate results, up to top {top_k}, for the data user provided. ",
        system_prompt=system_prompt,
        json_response=True,
        temperature=0.3,
    )

    algorithm_candidates = parse_response(response)
    state.algorithm.algorithm_candidates = algorithm_candidates

    # TODO：reason 尚未使用
    # If response is a list, get the first element and then access 'reasoning'
    if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and 'reasoning' in response[0]:
        log.info(f"algorithm selector reason: {response[0]['reasoning']}")
    elif isinstance(response, dict) and 'reasoning' in response:
        log.info(f"algorithm selector reason: {response['reasoning']}")
    else:
        log.info("algorithm selector reason: Not found in response")








    
    
    