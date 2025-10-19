#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置模块
提供数据库连接功能
"""

import os
import sys
import pymysql
from pymysql.cursors import DictCursor

# MySQL 配置
MYSQL_DB_PWD = os.getenv("MYSQL_DB_PWD", "root")
MYSQL_DB_USER = os.getenv("MYSQL_DB_USER", "root")
MYSQL_DB_HOST = os.getenv("MYSQL_DB_HOST", "localhost")
MYSQL_DB_PORT = int(os.getenv("MYSQL_DB_PORT", 3306))
MYSQL_DB_NAME = os.getenv("MYSQL_DB_NAME", "media_crawler")

mysql_db_config = {
    "user": MYSQL_DB_USER,
    "password": MYSQL_DB_PWD,
    "host": MYSQL_DB_HOST,
    "port": MYSQL_DB_PORT,
    "db_name": MYSQL_DB_NAME,
}

def get_db_conn():
    """
    获取数据库连接
    返回一个pymysql连接对象
    """
    try:
        # 使用配置的数据库参数
        conn = pymysql.connect(
            host=mysql_db_config['host'],
            user=mysql_db_config['user'],
            password=mysql_db_config['password'],
            database=mysql_db_config['db_name'],
            port=mysql_db_config['port'],
            charset='utf8mb4',
            cursorclass=DictCursor
        )
        print(f"✅ 数据库连接成功: {mysql_db_config['host']}:{mysql_db_config['port']}/{mysql_db_config['db_name']}")
        return conn
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print(f"配置信息: {mysql_db_config}")
        # 为了让程序能够继续运行，返回None
        # 在实际使用时会捕获异常
        return None