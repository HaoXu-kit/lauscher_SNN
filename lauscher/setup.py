# lauscher/setup.py

from setuptools import setup, find_packages

# 从 requirements.txt 文件中读取依赖项
with open('requirements_new.txt') as f:
    required = f.read().splitlines()

setup(
    name='lauscher',
    version='0.1.0',  # 你可以自己定义版本号
    description='A library for encoding audio into spikes.', # 简单的描述
    author='ujzax', # 换成你或原作者的名字
    packages=find_packages(), # ！！！这行代码会自动找到 lauscher/ 子文件夹作为要安装的包
    install_requires=required, # ！！！这行代码会自动处理所有依赖
    python_requires='>=3.9', # 指定支持的Python版本
)