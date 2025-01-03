import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
from multiprocessing import Manager

import tiktoken
import chromadb
import pdfplumber
from chromadb.utils import embedding_functions
from hg_rag.utils import split_text
import os
# from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from hg_rag import BAAIEmbeddingModel, SBertEmbeddingModel, QwenQAModel, QwenSummarizationModel

def increment_number(number):
    """Function to increment a number and add it to the shared list."""
    time.sleep(1)  # Simulate a time-consuming task
    return number + 1

def pdf_to_string(pdf_path):
    all_text = ""  # 用于存储所有页面的文本
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"  # 提取每页文本并换行
    return all_text

if __name__ == '__main__':
    # # Shared resource
    # shared_list = []
    #
    # # Lock for synchronizing access to the shared resource
    # lock = Lock()
    #
    # def increment_number(number):
    #     """Function to increment a number and add it to the shared list."""
    #     time.sleep(1)  # Simulate a time-consuming task
    #     with lock:
    #         shared_list.append(number + 1)
    #         print(f"Number {number} incremented to {number + 1}")
    #
    # # List of numbers to increment
    # numbers = [1, 2, 3, 4, 5]
    #
    #
    # t1 = time.time()
    # # Use ThreadPoolExecutor to run the increment_number function in multiple threads
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     executor.map(increment_number, numbers)
    #
    # t2 = time.time()
    # print("Final shared list:", shared_list)
    # print('Time taken:', t2-t1)

    # Manager to create a shared list
    # result = []


    # # List of numbers to increment
    # numbers = [1, 2, 3, 4, 5]
    #
    # t1 = time.time()
    # with ProcessPoolExecutor(max_workers=5) as executor:
    #     result = executor.map(increment_number, numbers)
    #
    # t2 = time.time()
    # print("Final shared list:", list(result))  # Convert shared list to a local list for printing
    # print('Time taken:', t2 - t1)


    # set the openai key in the env variable
    # pdf_path = 'demo/RS_zishi.pdf'
    # text1 = pdf_to_string(pdf_path=pdf_path)
    # text1 = text1[:10000]
    # # print(text1)

    # from hg_rag import RetrievalAugmentation, RetrievalAugmentationConfig
    # RAC = RetrievalAugmentationConfig(
    #     summarization_model=QwenSummarizationModel(model_name='/home/rt/data/model/Qwen/Qwen2.5-7B-Instruct'), 
    #     qa_model=QwenQAModel(model_name='/home/rt/data/model/Qwen/Qwen2.5-7B-Instruct'), 
    #     # embedding_model=BAAIEmbeddingModel(model_path='/home/rt/data/model/BAAI/bge-m3')
    #     embedding_model=SBertEmbeddingModel(model_name='/home/rt/data/model/sentence-transformers/multi-qa-mpnet-base-cos-v1')
    # )
    # RA = RetrievalAugmentation(config=RAC)
    # RA.add_documents(text1) # persist_path='./db3'
    # question = "Why the P3C have better performance than the EA ?"
    # context, layer_info = RA.retrieve(question=question, retrieve_mode='bottom_up', return_layer_information=True)
    # print(layer_info)
    # answer = RA.answer_question(question=question)
    # print("Answer: ", answer)

    a = None
    if a == None:
        print('yes')


    # emb_model = BAAIEmbeddingModel(model_path='/home/rt/data/model/BAAI/bge-m3')
    # ret = emb_model.create_embedding(text='i want to fuck you') # ['i want to fuck you', 'you are a bitch']
    # print(ret.shape)

    # emb_model1 =SBertEmbeddingModel(model_name='/home/rt/data/model/sentence-transformers/multi-qa-mpnet-base-cos-v1')
    # ret = emb_model1.create_embedding(text=['i want to fuck you'])
    # print(type(ret))

    # gen_model = QwenQAModel(model_name='/home/rt/data/model/Qwen/Qwen2.5-7B-Instruct')
    # ret = gen_model.answer_question(context='hello, i am Tao Ren', question='what is my name?')
    # print(ret)
