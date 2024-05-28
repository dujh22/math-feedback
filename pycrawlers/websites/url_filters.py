# -*- coding: UTF-8 -*-
# @Time : 2023/6/30 15:03 
# @Author : 刘洪波

photo_type = ['bmp', 'jpg','png', 'tif', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd',
              'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'ai', 'raw', 'WMF', 'webp', 'avif', 'apng']

video_type = ['wmv', 'asf', 'asx', 'rm', 'rmvb', 'mp4', '3gp', 'mov', 'm4v', 'avi',
              'dat', 'mkv', 'flv', 'vob', 'mpeg']


def filter_video_photo(url: str):
    all_types = photo_type + video_type
    for i in all_types:
        if url.endswith('.' + i):
            return False
    return True
