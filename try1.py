import os
from raptor import RetrievalAugmentation
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader

if __name__ == '__main__':
    # set the openai key in the env variable
    RA = RetrievalAugmentation()

    # a paper used for testing
    pdf_path = 'demo/RS_zishi.pdf'
    loader = PyPDFLoader(pdf_path)
    pdf_content = loader.load()
    text1 = ' '.join([page.page_content for page in pdf_content])
    text1 = text1[:10000]
    RA.add_documents(text1, persist_path='./db3')

    # # another testing example: Cinderella story defined in sample.txt
    # with open('demo/sample.txt', 'r') as file:
    #     text = file.read()
    # RA.add_documents(text, persist_path='./db3')

    question = "Why the P3C have better performance than the EA ?"
    # retrieve_mode = 'bottom_up' is our novel retrieve method! other method is not novel
    context, layer_info = RA.retrieve(question=question, retrieve_mode='bottom_up', return_layer_information=True)
    print(layer_info)

    # answer = RA.answer_question(question=question)
    # print("Answer: ", answer)
