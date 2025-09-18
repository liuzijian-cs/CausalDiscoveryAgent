# main.py
# Author: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# Last Update: 2025-09-18 19:00
# Description: main

import os
import logging
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

# def default_arguments():
    



def main():
    


    log.info("Hello from causaldiscoveryagent!")


if __name__ == "__main__":
    main()
