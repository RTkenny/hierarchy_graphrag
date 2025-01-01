import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
from multiprocessing import Manager

import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from raptor.utils import split_text
import os
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader


def increment_number(number):
    """Function to increment a number and add it to the shared list."""
    time.sleep(1)  # Simulate a time-consuming task
    return number + 1


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

    pdf_path = 'demo/RS_zishi.pdf'
    loader = PyPDFLoader(pdf_path)
    pdf_content = loader.load()
    text1 = ' '.join([page.page_content for page in pdf_content])
    text1 = text1[:10000]

    from raptor import RetrievalAugmentation
    RA = RetrievalAugmentation()
    RA.add_documents(text1, persist_path='./db3')
    question = "Why the P3C have better performance than the EA ?"
    context, layer_info = RA.retrieve(question=question, retrieve_mode='bottom_up', return_layer_information=True)
    print(layer_info)

    # answer = RA.answer_question(question=question)
    # print("Answer: ", answer)
