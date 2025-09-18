# CausalDiscoveryAgent
针对表格数据的因果发现系统。

开发中，初期版本主要基于工作流。

前端交互主要基于Streamlit

# Environment
本项目环境管理基于UV

## UV:
```bash
uv sync
```

## Anaconda:
```bash
conda create -n cda python=3.13
conda activate cda
pip install .
```

# 项目结构：
```text
CausalDiscoveryAgent/
├── .env.example                    # 环境变量模板
├── .gitignore
├── README.md
├── README.md

```

# 数据输入标准：
- 请首先移除id列；
- 默认最后一列为特征列；




# Acknowledgements & References:
This project is heavily inspired by [Causal-Copilot](https://github.com/Lancelot39/Causal-Copilot), and we sincerely appreciate its contributions to the field of causal discovery. Building on this work, our project introduces targeted optimizations for tabular data and focuses on delivering a primarily Chinese-language user experience.

