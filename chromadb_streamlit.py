import streamlit as st
import os
import tempfile
import copy
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import (
    LongContextReorder,
)

llm = ChatOpenAI(model="gpt-4-turbo")

print(100*'=')

with st.sidebar:
    pku_access_token = st.text_input("请输入您的测试号", key="chatbot_api_key", type="password")
    "[北京大学金融工程实验室](https://finlab.pku.edu.cn)"
    "[北京大学经济学院](https://econ.pku.edu.cn)"

    "\n"
    "【管理您的知识库】"

    pdf_files = st.file_uploader("上传您的PDF文件", accept_multiple_files=True, type="pdf")
    # if 'client' not in st.session_state:
    #     st.session_state.client = chromadb.PersistentClient(path='./db1')
    # client = st.session_state.client

    client = st.session_state.get('client', chromadb.PersistentClient(path='./db1'))
    default_ef = st.session_state.get('default_ef', embedding_functions.SentenceTransformerEmbeddingFunction(model_name='D:/python_data/model/BAAI/bge-large-zh-v1.5'))
    print('current collection:', client.list_collections())
    add_pdf_dict = st.session_state.get('add_pdf_dict', {})
    add_pdf_files = st.session_state.get("add_pdf_files", [])
    add_pdf_ids = st.session_state.get("add_pdf_ids", [])
    cur_pdf_id = st.session_state.get("cur_pdf_id", 0)
    for pdf_file in pdf_files:
        file_name = pdf_file.name
        if file_name in add_pdf_files:
            continue
        try:
            temp_file_name = None
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())
                st.markdown(f"Adding {file_name} to knowledge base...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                loader = PyPDFLoader(file_path=f.name)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                cur_pdf_id += 1
                collection = client.get_or_create_collection(name=f'doc{cur_pdf_id}', embedding_function=default_ef)
                collection.add(
                    documents=[doc.page_content for doc in docs],
                    ids=[f'ids{i}' for i in range(len(docs))],
                )

                st.markdown("")
                add_pdf_dict[cur_pdf_id] = file_name

                add_pdf_files.append(file_name)
                # add_pdf_ids.append(cur_pdf_id)

        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()

    file_name_list = [pdf_file.name for pdf_file in pdf_files]
    remove_id = []
    for key in add_pdf_dict.keys():
        if add_pdf_dict[key] not in file_name_list:
            remove_id.append(key)
    print('remove_id:', remove_id)
    print('add_pdf_dict', add_pdf_dict)
    print('add_pdf_files', add_pdf_files)
    for key in remove_id:
        print('file to be removed:', add_pdf_dict[key])
        add_pdf_files.remove(add_pdf_dict[key])
        client.delete_collection(f'doc{key}')
        del add_pdf_dict[key]


    # for ifile, file_name in enumerate(add_pdf_files):
    #     if file_name not in file_name_list:
    #         doc_id = add_pdf_ids[ifile]
    #         print(f'doc{doc_id}')
    #         client.delete_collection(f'doc{doc_id}')
    #
    #         add_pdf_files.remove(file_name)
            # add_pdf_ids.remove(doc_id)

    st.session_state["add_pdf_dict"] = add_pdf_dict
    st.session_state["add_pdf_files"] = add_pdf_files
    st.session_state["add_pdf_ids"] = add_pdf_ids
    st.session_state.cur_pdf_id = cur_pdf_id



# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

st.title("💬 金融AI助手")
st.caption("——北京大学金融工程实验室金融大模型科研项目")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
        您好，欢迎试用北京大学金融工程实验室开发的金融AI助手。 AI助手将基于金融工程实验室的数据库试图回答与金融财经相关的问题，并通过您的反馈不断学习!\n
        您可以就金融财经类学科问我任何问题，并帮助我成长。我会学得很快. :)
        """,
        }
    ]

for msg in st.session_state.messages:
    # 根据消息角色 (用户或助手) 创建一个对话框并显示消息内容
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    input_meg = copy.deepcopy(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户
    st.chat_message("user").write(prompt)

    if len(client.list_collections()) != 0:
        context = []
        for item in client.list_collections():
            collection0 = client.get_collection(item.name, embedding_function=default_ef)
            results = collection0.query(query_texts=prompt, n_results=5)
            print(results)
            context.append(format_docs(results["documents"][0]))
        context = format_docs(context)
        # context = retriever.invoke(prompt)
        # reordering = LongContextReorder()
        # context = reordering.transform_documents(context)
        # context = format_docs(context)
        print("with RAG")
        input_meg.append({"role": "user", "content": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "text",
                "text": f'the additional context for the conversation is {context}',
            },
        ]
        })
    else:
        input_meg.append({"role": "user", "content": prompt})
        print("without RAG")

    input_meg.append({"role": "user", "content": prompt})

    response = llm.invoke(input_meg)
    msg = response.content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    # print(len(st.session_state.messages))