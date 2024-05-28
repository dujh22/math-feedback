# -*- coding: UTF-8 -*-
# @Time : 2023/6/30 17:28 
# @Author : 刘洪波

from concurrent.futures import ThreadPoolExecutor
from pycrawlers.websites.get_web_page import crawl_web_page
from pycrawlers.websites.get_web_page_id import crawl_web_page_id
import time


def crawl(mongo_host: str, mongo_port: str, db_name: str, id_collection_name: str, collection_name: str,
          base_url: str, start_done: int = 0, end_done: int = 1, proxies=None, logger=None, url_filter=None):
    pool = ThreadPoolExecutor(max_workers=2)

    pool.submit(crawl_web_page_id, mongo_host, mongo_port, db_name, id_collection_name, collection_name,
                base_url, start_done, end_done, proxies, logger, url_filter)
    time.sleep(3)
    pool.submit(crawl_web_page, mongo_host, mongo_port, db_name, id_collection_name, collection_name,
                start_done, end_done, proxies, logger)
