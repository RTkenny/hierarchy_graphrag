import os
import json
import datasets
from tqdm import tqdm
# from memorag import Model
from functools import partial
from functools import partial
from transformers.utils import logging
from torch.utils.data import DataLoader
from longbench_utils import DATASET2CATEGORY, scorer, DATASET2PROMPT, DATASET2MAXNEWTOKENS, makedirs, FileLogger, DefaultDataCollator
from hg_rag import BAAIEmbeddingModel, SBertEmbeddingModel, QwenQAModel, QwenSummarizationModel

logger = logging.get_logger(__name__)

def process_longbench(data, indices, tokenizer, max_length=3500, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": []}

    for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):
        if dataset.endswith("_e"):
            dataset = dataset[:-2]

        if dataset in ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'qmsum']:
            question = input
        elif dataset == "gov_report":
            question = ""
        elif dataset == "multi_news":
            question = ""
        else:
            continue
        
        if max_length is not None:
            if truncate_from_middle:
                try:
                    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
                except:
                    tokenized_context = tokenizer.encode(context)
                if len(tokenized_context) > max_length:
                    half = int(max_length / 2)
                    context = tokenizer.decode(tokenized_context[:half]) + tokenizer.decode(tokenized_context[-half:])
            else:
                tokenized_context = tokenizer.encode(context)
                context = tokenizer.decode(tokenized_context[-max_length:])

        length = len(tokenizer.encode(context))

        outputs["context"].append(context)
        outputs["question"].append(question)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)
        outputs["length"].append(length)

    return outputs

if __name__ == '__main__':
    sum_model=QwenSummarizationModel(model_name='/home/rt/data/model/Qwen/Qwen2.5-7B-Instruct',load_in_4bit=True)
    qa_model=QwenQAModel(model_name='/home/rt/data/model/Qwen/Qwen2.5-7B-Instruct',load_in_4bit=True) 
    # embedding_model=BAAIEmbeddingModel(model_path='/home/rt/data/model/BAAI/bge-m3')
    emb_model=SBertEmbeddingModel(model_name='/home/rt/data/model/sentence-transformers/multi-qa-mpnet-base-cos-v1')

    output_dir = "./results/longbench/"

    dataset_names = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique'] # ['narrativeqa', 'qasper', 'hotpotqa'], ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique'] 
    # raw_dataset = datasets.load_dataset("json", data_files=f'/home/rt/data/MemoRAG/THUDM/LongBench/data/{dataset_names[0]}.jsonl', split="train")
    raw_dataset = datasets.load_dataset("json", data_files='../dataset/TommyChien/MemoRAG-data/longbench.json', split="train")

    max_length = 100000
    truncate_from_middle = True

    process_fn = partial(
                process_longbench, 
                tokenizer=qa_model.tokenizer,
                max_length=max_length,
                truncate_from_middle=truncate_from_middle
            )

    dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)
    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    result_dir = ''
    result_dir = os.path.join(output_dir, result_dir)

    for i, dataset_name in enumerate(dataset_names):
        logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")
        
        dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)

        data_collator = DefaultDataCollator(padding_side="left")
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=data_collator,
            # only pin memory when no gpu
        )

        indices = []
        preds = []
        memory_results = []
        _prompt = DATASET2PROMPT[dataset_name]
        task_max_new_token=DATASET2MAXNEWTOKENS[dataset_name]
        
        for i, x in enumerate(tqdm(dataloader, desc="Generating")):
            x.pop("dataset")
            index = x.pop("index")[0]

            # # generate output
            # prompt = _prompt.format(context=x["context"][0], input=x["question"][0])

            from hg_rag import RetrievalAugmentation, RetrievalAugmentationConfig
            RAC = RetrievalAugmentationConfig(
                summarization_model=sum_model,
                qa_model=qa_model, 
                # embedding_model=BAAIEmbeddingModel(model_path='/home/rt/data/model/BAAI/bge-m3')
                embedding_model=emb_model,
                tb_num_layers=1,
                tb_max_tokens=600,
                tb_summarization_length=60,
            )
            RA = RetrievalAugmentation(config=RAC)
            RA.add_documents(x["context"][0]) # persist_path='./db3'
            output = RA.answer_question(question=x["question"][0], prompt_template=_prompt, top_k=30, gen_max_tokens=task_max_new_token, retrieve_mode='bottom_up')
            # output = gen_model.generate(prompts=prompt, max_new_tokens=task_max_new_token, do_sample=True)

            print(output)
            output = [output]

            index = index.tolist()
            preds.extend(output)
            if isinstance(index, list):
                indices.extend(index)
            else:
                # single process
                indices.append(index)

            raw_dataset_subset = raw_dataset[indices]
            answers = raw_dataset_subset["answers"]
            lengths = raw_dataset_subset["length"]
            all_classes = []
            score = scorer(dataset_name, preds, answers, all_classes)        
            
            logger.info(f"{dataset_name}: {score}")
            metrics[dataset_name] = score

            with open(makedirs(result_path), "w", encoding="utf-8") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for index, pred in zip(indices, preds):
                    sample = raw_dataset[index]
                    del sample["context"]
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")