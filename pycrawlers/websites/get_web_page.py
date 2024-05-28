# -*- coding: UTF-8 -*-
# @Time : 2023/6/21 16:46 
# @Author : 刘洪波
import time
from pycrawlers.common.tools import MongoHelper, requests_get, sleep_some_time


def crawl_web_page(mongo_host: str, mongo_port: str, db_name: str, id_collection_name: str, collection_name: str,
                   start_done: int = 0, end_done: int = 1, proxies=None, logger=None):
    """
    获取网页
    :param mongo_host:
    :param mongo_port:
    :param db_name:
    :param id_collection_name: 存储网页url的表名
    :param collection_name: 存储网页的表名
    :param start_done: 未抓取的url标记
    :param end_done: 已抓取的url标记
    :param proxies: 代理
    :param logger: 日志
    :return:
    """
    mg_hp = MongoHelper(mongo_host, mongo_port)
    id_collection = mg_hp.get_collection(id_collection_name, db_name)
    collection = mg_hp.get_collection(collection_name, db_name)
    if logger:
        logger.info('开始抓取内容')
    else:
        print('开始抓取内容')
    count = 0
    start_time = time.time()
    while True:
        with id_collection.find({"done": start_done}, no_cursor_timeout=True, batch_size=10) as cursor:
            for i in cursor:
                _id = i['_id']
                count += 1
                if collection.find_one({'_id': _id}):
                    id_collection.update_one({'_id': _id}, {'$set': {'done': end_done}})
                    continue
                response = requests_get(_id, is_etree=False, timeout=20, proxies=proxies, logger=logger)
                if response != 'an error occurred':
                    try:
                        collection.insert_one({'_id': _id, 'html_text': response.text, 'done': 0})
                        id_collection.update_one({'_id': _id}, {'$set': {'done': end_done}})
                    except Exception as e:
                        if logger:
                            logger.info(e)
                        else:
                            print(e)
                sleep_some_time(count, start_time, sleep_key_num=50)
        time.sleep(360)
        info_str = '已经抓取数量：' + str(count)
        if logger:
            logger.info(info_str)
        else:
            print(info_str)
