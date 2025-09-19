# main.py
# Author: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# Last Update: 2025-09-18 19:00
# Description: main



import os
import logging
import argparse
import warnings

from dotenv import load_dotenv

from src.core import Initializer

# Environment variables
load_dotenv()

# 抑制不重要的警告
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message=".*font.*")
warnings.filterwarnings("ignore", message=".*Failed to extract font.*")

# 设置 matplotlib 和其他库的日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Logger:
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def default_arguments():
    parser = argparse.ArgumentParser(description='Causal Discovery Agent.')
    parser.add_argument('--data', type=str, default="demo/data/data.csv",
                        help='Path to the input dataset file (e.g., CSV format or Excel file)')
    return parser.parse_args()



def main():
    initializer = Initializer(data_path="demo/data/data.csv")
    state = initializer.get_state()

    from src.causal_discovery.wrappers.pc import PC
    PC_model = PC(state, params={})
    if state.user_data.processed_data is None:
        log.error("Processed data is None. Please check your data preprocessing pi" \
        "" \
        "peline.")
        return
    adj_matrix, info = PC_model.fit(state.user_data.processed_data)
    print("Adjacency Matrix:\n", adj_matrix)
    print("Info:\n", info)
    

    # print(initializer.get_available_alorithms())

    # print(state)
    # print(state.algorithm.algorithm_candidates)



    log.info("Hello from causaldiscoveryagent!")


if __name__ == "__main__":
    main()
