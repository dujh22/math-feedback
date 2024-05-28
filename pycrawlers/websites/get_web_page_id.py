# -*- coding: UTF-8 -*-
# @Time : 2023/6/21 16:47 
# @Author : 刘洪波

import time
from lxml import etree
from pycrawlers.common.tools import MongoHelper, requests_get


def crawl_web_page_id(mongo_host: str, mongo_port: str, db_name: str, id_collection_name: str, collection_name: str,
                      base_url: str, start_done: int = 0, end_done: int = 1, proxies=None, logger=None,
                      url_filter=None):
    """
    获取网页url
    :param mongo_host:
    :param mongo_port:
    :param db_name:
    :param id_collection_name: 存储网页url的表名
    :param collection_name: 存储网页的表名
    :param base_url: 网页首页
    :param start_done: 未抓取的url标记
    :param end_done: 已抓取的url标记
    :param proxies: 代理
    :param logger: 日志
    :param url_filter: url 过滤器, 一个函数，输入url 判断是否需要抓取， 返回值 True 或 Fasle, True表示需要抓取
    :return:
    """
    mg_hp = MongoHelper(mongo_host, mongo_port)
    id_collection = mg_hp.get_collection(id_collection_name, db_name)
    collection = mg_hp.get_collection(collection_name, db_name)
    if not collection.find_one({'_id': base_url}):
        response = requests_get(base_url, is_etree=False, timeout=20, proxies=proxies, logger=logger)
        if response != 'an error occurred':
            collection.insert_one({'_id': base_url, 'html_text': response.text, 'done': 0})
        else:
            raise Exception('首页抓取失败！！！请检查网络或代理后重试！！！')
    if logger:
        logger.info('开始获取id')
    else:
        print('开始获取id')
    while True:
        with collection.find({'done': start_done}, no_cursor_timeout=True, batch_size=1000) as cursor:
            for i in cursor:
                html_text = i['html_text']
                _id = i['_id']
                if html_text:
                    try:
                        site_html = etree.HTML(html_text)
                        for href in site_html.xpath('//a/@href'):
                            if href:
                                if base_url in href:
                                    url = href
                                else:
                                    if href[0] != '/':
                                        continue
                                    elif len(href) > 1 and href[1] == '/':
                                        url = 'https:' + href
                                        if base_url not in url:
                                            continue
                                    else:
                                        url = base_url + href
                                tag = True
                                if url_filter:
                                    tag = url_filter(url)
                                if tag:
                                    if id_collection.find_one({'_id': url}):
                                        continue
                                    id_collection.insert_one({'_id': url, 'done': 0})
                    except Exception as e:
                        if logger:
                            logger.info(e)
                        else:
                            print(e)
                else:
                    info_str = 'no html data :' + str(_id)
                    if logger:
                        logger.info(info_str)
                    else:
                        print(info_str)
                collection.update_one({'_id': _id}, {'$set': {'done': end_done}})
        time.sleep(360)
