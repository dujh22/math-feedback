# -*- coding: UTF-8 -*-
# @Time : 2022/8/17 15:40 
# @Author : 刘洪波
import requests
from requests.adapters import HTTPAdapter
from functools import wraps
from tqdm import tqdm
import os
from lxml import etree
import time
import random
from pymongo import MongoClient
from pytz import timezone
import json
from pycrawlers.common.default_data import default_headers


def get_session(max_retries: int = 3):
    """
    使用requests Session，使抓取数据的时候可以重试
    # 默认设置重试次数为3次
    """
    session = requests.Session()
    session.mount('http://', HTTPAdapter(max_retries=max_retries))
    session.mount('https://', HTTPAdapter(max_retries=max_retries))
    return session


class DealException(object):
    """处理异常返回的装饰器"""
    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            try:
                response = func(*args, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                print(e)
        return wrapped_function


# 下载数据
def download(url: str, fname: str, headers: dict, read_timeout: int = 15, file_size=0, max_retries: int = 3):
    if 'Range' in headers:
        del headers['Range']

    requests_session = get_session(max_retries)

    @DealException()
    def get_data():
        return requests_session.get(url, headers=headers, stream=True, timeout=(read_timeout, 5))
    resp_ = get_data()
    total_ = int(resp_.headers.get('content-length', 0))
    if file_size < total_:
        file_op = 'wb'
        if file_size:
            headers['Range'] = f'bytes={file_size}-'
            file_op = 'ab'
        time.sleep(random.random())
        resp = get_data()
        # total = int(resp.headers.get('content-length', 0))
        with open(fname, file_op) as file, tqdm(
            desc=fname,
            total=total_,
            initial=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print(fname, ' ✅')
        time.sleep(random.randint(1, 3))


def juedge_path(file_path: str):
    """判断路径是否存在，不存在就创建"""
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if file_path[-1] != '/':
        file_path += '/'
    return file_path


def juedge_url(url: str):
    """url检查"""
    if 'http://' not in url and 'https://' not in url:
        raise ValueError(f'URL error, the url is missing http or https, your url is {url}')


def deal_response(response, is_etree: bool = False):
    if response.encoding != 'utf-8':
        response.encoding = 'utf-8'
    if response.status_code == 200:
        if is_etree:
            res_html = etree.HTML(response.content)
            return res_html
        else:
            return response
    else:
        return ''


def sleep_some_time(count: int, start_time, report_key_num: int = 1000,
                    sleep_key_num: int = 10):
    if count % sleep_key_num == 0:
        time.sleep(random.randint(3, 5))
    else:
        time.sleep(random.random())
    if count % report_key_num == 0:
        print(f'耗时：' + str(time.time() - start_time))
        print(f'获取数据的数量：' + str(count))
        print('\n')


class MongoHelper(object):
    """MongoDB 连接"""

    def __init__(self, host, port, database=None, user=None, password=None):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.uri = f"mongodb://{self.host}:{self.port}"
        self.client = MongoClient(self.uri, tz_aware=True, tzinfo=timezone('Asia/Shanghai'))

    def get_database(self, database=None):
        if not database:
            database = self.database
        if self.user and self.password:
            self.client[database].authenticate(self.user, self.password)
        return self.client[database]

    def get_collection(self, collection_name, database=None):
        _db = self.get_database(database=database)
        return _db[collection_name]


def requests_get(url: str, is_etree: bool = False, headers: dict = None,
                 timeout: int = 10, proxies=None, logger=None):
    if not headers:
        headers = default_headers
    try:
        if proxies:
            response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        else:
            response = requests.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        if logger:
            logger.info(e)
        else:
            print(e)
        return 'an error occurred'
    return deal_response(response, is_etree)


def requests_post(url: str, data: dict, is_etree: bool = False, headers: dict = None,
                  timeout: int = 10, proxies=None, logger=None):
    if not headers:
        headers = default_headers
    try:
        if proxies:
            response = requests.post(url, data=json.dumps(data), headers=headers, timeout=timeout, proxies=proxies)
        else:
            response = requests.post(url, data=json.dumps(data), headers=headers, timeout=timeout)
    except Exception as e:
        if logger:
            logger.info(e)
        else:
            print(e)
        return 'an error occurred'
    return deal_response(response, is_etree)
