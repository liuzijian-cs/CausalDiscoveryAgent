# main.py
# Author: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# Last Update: 2025-09-18 19:00
# Description: main

import os
import logging
import argparse

from dotenv import load_dotenv

# Environment variables
load_dotenv()

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



    log.info("Hello from causaldiscoveryagent!")


if __name__ == "__main__":
    main()
