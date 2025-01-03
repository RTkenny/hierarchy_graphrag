import logging
from abc import ABC, abstractmethod
import os
from openai import OpenAI
from typing import List, Mapping, Optional, Union
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache, pipeline
import getpass
import torch
from minference import MInference
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# embedding model
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text, show_progress_bar=False)
    

class BAAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_path='', dense_metric='cos', pooling_method:List[str]=["cls"], dtype='fp16', load_in_4bit=False):
        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.dense_metric = dense_metric
        self.pooling_method = pooling_method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path, torch_dtype=dtype, device_map={'': "cuda"}, load_in_4bit=load_in_4bit).eval()

    @property
    def device(self):
        return self.encoder.device

    def _prepare(self, inputs):
        inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, truncation=True, max_length=100000)
        inputs = inputs.to(self.device)
        return inputs
    
    def _pool(self, embeddings):
        if "cls" in self.pooling_method:
            embedding = embeddings[:, 0]
        else:
            raise NotImplementedError(
                f"Pooling_method {self.pooling_method} not implemented!")
        return embedding

    def create_embedding(self, text: Union[str, List[str], Mapping]):
        inputs = self._prepare(text)
        encoder = self.encoder

        embeddings = encoder(**inputs).last_hidden_state    # B, L, D
        embeddings = self._pool(embeddings)
        if self.dense_metric == "cos":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()


# QA model
class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question, prompt_template=None):
        pass


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, prompt_template=None, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e

class QwenQAModel(BaseQAModel):
    # def __init__(
    #     self,
    #     model_name="Qwen/Qwen2.5-7B-Instruct",
    #     cache_dir: str="",
    #     access_token: str="",
    #     beacon_ratio: int=None,
    #     load_in_4bit: bool=False,
    #     enable_flash_attn: bool=False
    # ):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     if enable_flash_attn:
    #             attn_implementation = "flash_attention_2"
    #     else:
    #         attn_implementation = None

    #     self.model_kwargs = {
    #         "cache_dir": cache_dir,
    #         "token": access_token,
    #         "device_map": {"": self.device},
    #         "attn_implementation": attn_implementation,
    #         "torch_dtype": torch.bfloat16,
    #         "trust_remote_code": True,
    #     }
    #     self.model_name = model_name

    #     if load_in_4bit:
    #         quant_config = BitsAndBytesConfig(
    #                 load_in_4bit=load_in_4bit
    #             )
    #         self.model_kwargs["quantization_config"] = quant_config


    #     tokenizer_kwargs = {
    #         "cache_dir": cache_dir,
    #         "token": access_token,
    #         "padding_side": "left",
    #         "trust_remote_code": True,
    #     }

    #     self.tokenizer = AutoTokenizer.from_pretrained(
    #         model_name, 
    #         **tokenizer_kwargs
    #     )

    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         model_name, 
    #         **self.model_kwargs
    #     ).eval()

    # def run_model(self, input_string, **generator_args):
    #     input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
    #         self.device
    #     )
    #     res = self.model.generate(input_ids, **generator_args)
    #     return self.tokenizer.batch_decode(
    #             res[:, input_ids.shape[1]:], 
    #             skip_special_tokens=True
    #         )

    # def answer_question(self, context, question):
    #     input_string = question + " \\n " + context
    #     output = self.run_model(input_string)
    #     return output[0]
    
    def __init__(self, model_name= "Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False):
        # Initialize the tokenizer and the pipeline for the model
        device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # Use "cpu" if CUDA is not available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": device},
        }
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit
                )
            self.model_kwargs["quantization_config"] = quant_config
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs=self.model_kwargs,
            # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def answer_question(self, context, question, prompt_template=None, gen_max_tokens=150):
        # Apply the chat template for the context and question
        if prompt_template==None:
            messages=[
                {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
            ]
        else:
            messages=[{'role':'user', 'content': prompt_template.format(context=context, input=question)}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the answer using the pipeline
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=gen_max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated answer
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]

# summarize model
class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

class QwenSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False):
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use "cpu" if CUDA is not available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_kwargs = {
            "device_map": {"": device},
            "torch_dtype": torch.bfloat16,
        }
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit
                )
            self.model_kwargs["quantization_config"] = quant_config
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs=self.model_kwargs,
        )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary



class Model:
    def __init__(
        self, 
        model_name_or_path: str, 
        cache_dir: str="",
        access_token: str="",
        beacon_ratio: int=None,
        load_in_4bit: bool=False,
        enable_flash_attn: bool=True
    ):  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if enable_flash_attn:
            if model_name_or_path.find("mistral") != -1:
                attn_implementation = "sdpa"
            else:
                attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None

        if model_name_or_path.find("memorag") == -1:
            load_in_4bit = True

        self.model_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "device_map": {"": device},
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        self.model_name_or_path = model_name_or_path

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit
                )
            self.model_kwargs["quantization_config"] = quant_config

        if beacon_ratio and model_name_or_path.find("memorag") != -1:
            self.model_kwargs["beacon_ratio"] = [beacon_ratio]

        tokenizer_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "padding_side": "left",
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            **tokenizer_kwargs
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **self.model_kwargs
        ).eval()

        print(f"Model loaded from {model_name_or_path}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def ids2text(
        self, 
        inputs, 
        **generation_kwargs
    ) -> str:
        outputs = self.model.generate(
            **inputs, 
            **generation_kwargs, 
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded_output = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

        return decoded_output

    def template2ids(
        self, 
        templates: List, 
        remove_symbol=None
    ):
        if isinstance(templates, str):
            templates = [templates]
        
        batch_prompts = []
        for template in templates:
            to_encode = self.tokenizer.apply_chat_template(
                template, 
                tokenize=False, 
                add_generation_prompt=True
            )
            if remove_symbol:
                to_encode = to_encode.replace(remove_symbol, "")
            batch_prompts.append(to_encode)

        inputs = self.tokenizer(
            batch_prompts, 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        return inputs

    def minference_patch(self, model_type:str="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        minference_patch = MInference("minference", model_type)
        self.model=minference_patch(self.model)

    def reload_model(self):
        # TODO 
        del self.model
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs
        ).eval()

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        batch_size: int = 1, 
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        repetition_penalty:float=1.0
    ) -> Union[str, List[str]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        }
            
        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = []
            for prompt in prompts[i: i + batch_size]:
                batch_prompts.append([{"role": "user", "content": prompt}])
            inputs = self.template2ids(batch_prompts)
            outputs = self.ids2text(inputs, **generation_kwargs)
            all_outputs.extend(outputs)
        return all_outputs