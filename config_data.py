
md5_path = "./md5.txt"  #md5字符串存储文件路径

#chroma
collection_name = "rag"  #数据库的表名
persist_directory = "./chroma_db"  #数据库本地存储文件夹

#spliter
chunk_size = 1000  #分割后每个文本段的最大长度
chunk_overlap = 100  #分割后每个文本段的重叠长度
separators = ["\n\n", "\n", ".","!","?","。","！","？",""," "]  #自然段落划分符号的分隔符列表
max_split_length = 1000

#相似度检索
similarity_search = 1  #返回的文档数量，默认是1


embedding_model_name = "text-embedding-v4"
chat_model = "qwen3-max"

session_config ={
        "configurable":{
            "session_id":"user_001"
        }
    }
