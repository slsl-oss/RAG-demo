from langchain_chroma import Chroma
import config_data as config

"""
获得检索器对象方便加入chain
检索器通过invoke方法调用，用户输入问题，返回检索结果

"""
class VectorStoreService(object):
    def __init__(self,embedding):
       """
       :param embedding: 向量嵌入模型
       """
       self.embedding = embedding

       self.vector_store = Chroma(
           collection_name=config.collection_name,
           embedding_function = self.embedding,
           persist_directory = config.persist_directory

       )


    def get_retriever(self):
        """
        :return: 放回向量检索器对象方便加入chain
        :param self:

         """
        return self.vector_store.as_retriever(search_kwargs={"k" : config.similarity_search})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    vector_store_service = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4"))
    retriever = vector_store_service.get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)




