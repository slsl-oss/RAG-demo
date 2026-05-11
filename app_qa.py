import streamlit as st
import time
from rag import RagService
import config_data as config


#设置一个标题(智能客服
st.title("智能客服")
st.divider()   #分隔符

#存储对话信息，页面刷新时会保留对话信息

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，我是智能客服，有什么我可以帮助你的吗？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()
#在页面输出对话信息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

#在页面最下方提供用户输入烂
prompt =  st.chat_input()


if prompt:
    #在页面输出用户提问
    st.chat_message("user").write(prompt)
    #将用户提问添加到对话信息中
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("AI思考中...."):
        res_stream = st.session_state["rag"].chain.stream({"input":prompt},config.session_config)
        #yield


        ai_res_list = []
        def capture_res(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture_res(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)}) #将ai_res_list中的字符串拼接起来，添加到对话信息中

