"""
知识库
"""
import hashlib
import os
from datetime import datetime

from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_md5(md5_str:str):
    """
    检查传入的md5字符串是否已经被处理
    return False: 未处理过
    return True: 已处理过
    """
    if not os.path.exists(config.md5_path):
        #if能进入说明文件不存在 --> 肯定没有处理过
        open(config.md5_path, 'w',encoding="utf-8").close #创建文件
        return False
    else:
        #if不能进入说明文件存在 --> 肯定有处理过
        with open(config.md5_path, 'r',encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() == md5_str:
                    return True
        return False

def save_md5(md5_str:str):
    """将传入的md5字符串，记录到文件中保存"""
    with open(config.md5_path, 'a',encoding="utf-8") as f:
        f.write(md5_str+"\n")

def get_string_md5(input_str:str,encoding="utf-8"):
    """获取字符串的md5值"""

    #将字符串转化为字节数组
    byte_str = input_str.encode(encoding)

    #创建md5对象
    md5_obj = hashlib.md5()

    #将字节数组传入md5对象
    md5_obj.update(byte_str)

    #获取md5值
    md5_hex = md5_obj.hexdigest()

    return md5_hex



class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在，创建文件夹
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,      #数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),      #嵌入函数
            persist_directory=config.persist_directory   #数据库本地存储文件夹

        )     #向量存储的是咧 Chroma向量数据库对象

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,         #分割后每个文本段的最大长度
            chunk_overlap=config.chunk_overlap,    #分割后每个文本段的重叠长度
            separators=config.separators,          #自然段落划分符号的分隔符列表
            length_function=len,                      #计算文本长度的函数
        )     #文本分割器对象


    def upload_by_str(self,data:str,filename):
        """
        将传入的字符串，进行向量化，存入数据库中
        :param data: 要向量化的字符串
        :param filename: 文件名
        :return:
        """
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.max_split_length:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]


        metadata = {
            "source":filename,
            #2026-01-01 00:00:00
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"PQ"
        }

        #内容加载到数据库中
        self.chroma.add_texts(
            #iterable : list,tuple,set 都是可迭代对象
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        #记录处理过的md5值
        save_md5(md5_hex)

        return "[成功]内容已经加载到知识库中"

if __name__ == '__main__':
    service = KnowledgeBaseService()
    r =service.upload_by_str("周杰伦是一个歌手,1999年1月1日出生,来自中国北京。","testfile")
    print(r)



