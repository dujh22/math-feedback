# -*- coding: UTF-8 -*-
# @Time : 2022/8/17 15:38 
# @Author : 刘洪波

"""
获取hugging face 模型
"""
import base64
import random
import os
import time
import requests
from lxml import etree
from pycrawlers.common.default_data import headers
from pycrawlers.common.tools import get_session
from pycrawlers.common.tools import DealException
from pycrawlers.common.tools import download
from pycrawlers.common.tools import juedge_path
from pycrawlers.common.tools import juedge_url


class HuggingFace(object):
    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = base_url if base_url else 'https://huggingface.co'
        self.tag = True if base_url else False
        self.html_data = None
        self.requests_session = get_session()
        self.headers = headers.copy()
        if token:
            self.headers['Cookie'] = f'token={token}'

    def get_data(self, url: str, file_save_path: str = None, max_retries: int = 3):
        """
        获取单个数据
        :param url: 例：'https://huggingface.co/albert-base-v2/tree/main'
        :param file_save_path: None or './albert-base-v2'
        :param max_retries: request 最大重试次数
        :return:
        """
        juedge_url(url)
        _url = url.split('/')
        if not self.tag:
            self.get_base_url(_url)
        main_id = _url.index('main')
        if len(_url) == (main_id + 1):
            item_name = _url[-3]
        else:
            item_name = _url[-1]
        files_path = juedge_path(file_save_path) if file_save_path else juedge_path('./' + item_name + '/')
        print(f"{'🔴' * 10}{' ' * 5}Start downloading: {item_name}{' ' * 5}{'🔴' * 10}")
        self.get_one_data(url, files_path, max_retries)
        print(f"{'🟢' * 10}{' ' * 5}Download completed{' ' * 5}{'🟢' * 10}")

    def get_one_data(self, url: str, files_path: str, max_retries: int = 3):
        """
        获取单个数据
        :param url: 例：'htt://huggingface.co/albert-base-v2/tree/main'
        :param files_path: 例：'./albert-base-v2'
        :param max_retries: request 最大重试次数
        :return:
        """
        juedge_url(url)
        response = self.crawl_html(url)
        if response:
            self.html_data = etree.HTML(response.content)
            # print(self.html_data)
            file_names = self.get_file_names()
            file_urls, dir_dict = self.get_file_urls()
            # print(file_urls)
            # print(file_names)
            # print(dir_dict)
            for dir_name, dir_url in dir_dict.items():
                self.get_one_data(dir_url, juedge_path(files_path + dir_name), max_retries)
            last_name, last_url = file_names[-1], file_urls[-1]
            if last_name not in last_url:
                raise ValueError('last_name does not match last_url, please check the page')
            dir_name = self.get_dir(url)
            base_crawl_url = last_url.split(last_name)
            file_urls_2, file_names_2 = [], []
            if self.load_more_files():
                file_urls_2, file_names_2 = self.get_more_files_info(url, last_name, base_crawl_url, dir_name)
                file_urls += file_urls_2
                file_names += file_names_2
            self.get_files(file_names, file_urls, files_path, max_retries)
            # 循环 load more files
            # 不使用递归处理的原因是：若获取了全部的待下载的 file_names与 file_urls后中断下载，在这种的情况下会资源浪费
            while file_urls_2 and file_names_2:
                file_urls_2, file_names_2 = self.get_more_files_info(url, file_names_2[-1], base_crawl_url, dir_name)
                if file_urls_2 and file_names_2:
                    self.get_files(file_names_2, file_urls_2, files_path, max_retries)

    def get_batch_data(self, urls: list, file_save_paths: list = None, count_info=True, max_retries: int = 3):
        """
        批量获取数据
        :param urls: ['https://huggingface.co/albert-base-v2/tree/main',
                      'https://huggingface.co/dmis-lab/biosyn-sapbert-bc5cdr-disease/tree/main']
        :param file_save_paths:['./model_1/albert-base-v2', './model_2/']
        :param count_info: 是否生成程序执行的统计信息
        :param max_retries: request 最大重试次数
        :return:
        """
        success_urls = []
        fail_urls = []
        if file_save_paths:
            if len(urls) == len(file_save_paths):
                for u, f in zip(urls, file_save_paths):
                    success_urls, fail_urls = self.fault_tolerant(u, success_urls, fail_urls, f, max_retries)
            else:
                raise ValueError('The number of urls and paths is inconsistent')
        else:
            for url in urls:
                success_urls, fail_urls = self.fault_tolerant(url, success_urls, fail_urls, max_retries=max_retries)
        if count_info:
            if success_urls or fail_urls:
                self.count_info(success_urls, fail_urls)

    def fault_tolerant(self, url: str, success_urls: list, fail_urls: list, path: str = None, max_retries: int = 3):
        """容错处理"""
        try:
            self.get_data(url, path, max_retries)
            success_urls.append(url)
            time.sleep(0.5)
        except Exception as e:
            print(e)
            fail_urls.append(url)
        return success_urls, fail_urls

    def get_base_url(self, _url: list):
        """获取基础url"""
        if len(_url) > 5:
            if 'http' in _url[0] and _url[1] == '':
                self.base_url = _url[0] + '//' + _url[2]

    @DealException()
    def crawl_html(self, url):
        """获取html"""
        return self.requests_session.get(url, headers=self.headers, timeout=1)

    def get_file_names(self):
        """获取文件名"""
        xpath = f'//div[@data-target="ViewerIndexTreeList"]/ul/li/div[1]/a[1]/span[1]/text()'
        return self.html_data.xpath(xpath)

    def get_file_urls(self):
        """获取文件链接"""
        xpath = f'//div[@data-target="ViewerIndexTreeList"]/ul/li/a[1]/@href'
        file_urls = self.html_data.xpath(xpath)
        new_file_urls, dir_dict = [], {}
        for u in file_urls:
            if u.endswith('?download=true'):
                new_file_urls.append(u)
            else:
                dir_dict[u.split('/')[-1]] = self.base_url + u
        return new_file_urls, dir_dict

    def load_more_files(self):
        """判断是否 Load more files"""
        xpath = f'//div[@data-target="ViewerIndexTreeList"]/ul/li/button/text()'
        _bt = self.html_data.xpath(xpath)
        bt = [t.strip() for t in _bt]
        if 'Load more files' in bt:
            return True
        return False

    def get_more_files_info(self, url: str, last_name: str, base_crawl_url: list, dir_name: str):
        """获取 more files 的文件名与url"""
        x = '{"file_name":"' + dir_name + last_name + '"}'
        y = base64.b64encode(x.encode())
        cursor = base64.b64encode(y + b':50')
        params = {'cursor': cursor, 'expand': True}
        new_url = url.replace('https://huggingface.co/', 'https://huggingface.co/api/')
        resp = requests.get(url=new_url, params=params, headers=self.headers)
        file_urls, file_names = [], []
        if resp.status_code in [200, 304]:
            resp_js = resp.json()
            for r in resp_js:
                file_name = r.get('path')
                if file_name:
                    if '/' in file_name:
                        file_name = file_name.replace(dir_name, '')
                    file_names.append(file_name)
                    file_urls.append(base_crawl_url[0] + file_name + base_crawl_url[1])
        return file_urls, file_names

    @staticmethod
    def get_dir(url: str):
        _dir = ''
        if '/main' in url:
            u = url.split('/main')[-1]
            if u:
                if u[0] == '/':
                    if len(u) > 1:
                        _dir = u[1:]
                else:
                    _dir = u
        if _dir and not _dir.endswith('/'):
            _dir += '/'
        return _dir

    @staticmethod
    def generate_file_path(_url: list):
        """生成路径"""
        return juedge_path('./' + _url[-3] + '/')

    def get_files(self, file_names, file_urls, files_path, max_retries):
        for name, part_url in zip(file_names, file_urls):
            if name in part_url:
                url = self.base_url + part_url
                save_file_path = files_path + name
                download(url, save_file_path, self.headers,
                         read_timeout=60, file_size=self.get_file_size(save_file_path), max_retries=max_retries)
                time.sleep(random.random())

    @staticmethod
    def count_info(success_urls, fail_urls):
        print(f'程序执行统计：')
        print(f'a. 成功{str(len(success_urls))}个')
        print(f'b. 失败{str(len(fail_urls))}个')
        print(f'c. 成功的URL：{"，".join(success_urls)}')
        print(f'd. 失败的URL：{"，".join(fail_urls)}')

    @staticmethod
    def get_file_size(file_path):
        """
        获取文件大小
        :param:  file_path:文件路径（带文件名）
        :return: file_size：文件大小
        """
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        else:
            return 0
