from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

import config_data as config
from vector_stores import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from file_history_store import get_history

class RagService(object):
    def __init__(self):

        self.vector_service  = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的资料为主简洁和专业的回答用户的问题，参考资料：{context}"),
                ("system","并且我提供用户对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model)

        self.chain = self.__get_chain()


    def __get_chain(self):
        #私有的方法获得执行链chain
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            format_str = ""
            for doc in docs:
                format_str += f"文档片段:{doc.page_content}\n元数据:{doc.metadata}\n\n"

            return format_str

        def print_prompt(prompt):
            print("="*20)
            print(prompt.to_string())
            print("="*20)

            return prompt

        def format_for_retriever(value):   #retriever:需要用户输入字符串，而此时的输入是一个字典
            return value["input"]

        def format_for_prompt(value):
            new_value = {}
            new_value["history"] = value["input"]["history"]
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            return new_value
        chain = (
            {
                "input":RunnablePassthrough(),    # 相当于用户输入的占位符
                "context": RunnableLambda(format_for_retriever) | retriever | format_document   #用户输入传入retriever检索器，检索chroma向量数据库,
                                                        # 结果是list[Document] 通过定义一个format_document函数转为字符串
            } | RunnableLambda(format_for_prompt) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history", #历史消息的占位
        )

        return conversation_chain


if __name__ == '__main__':
    #session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001"
        }
    }
    rag_service = RagService()
    res = rag_service.chain.invoke({"input":"我身高170cm，尺码推荐"},session_config)
    print(res)
