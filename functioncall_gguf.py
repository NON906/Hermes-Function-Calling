import os
import sys
import json
import argparse

from huggingface_hub import hf_hub_download

from langchain.prompts import StringPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.tool import ToolMessage
from langchain_community.llms import LlamaCpp
from langchain_community.chat_message_histories import ChatMessageHistory

from gpt_stream_parser import force_parse_json

from functioncall import ModelInference
from prompter import PromptManager
from utils import (
    inference_logger,
    validate_and_extract_tool_calls,
    get_assistant_message,
)


class TemplateMessagesPrompt(StringPromptTemplate):
    history_name: str = 'history'

    def format(self, **kwargs: any) -> str:
        input_mes_list = kwargs[self.history_name]
        messages = ''
        for mes in input_mes_list:
            messages += '<|im_start|>'
            if type(mes) is tuple:
                messages += mes[0] + '\n' + mes[1] + '<|im_end|>\n'
            else:
                if type(mes) is HumanMessage:
                    messages += 'user'
                elif type(mes) is AIMessage:
                    messages += 'assistant'
                elif type(mes) is SystemMessage:
                    messages += 'system'
                else:
                    messages += 'tool'
                messages += '\n' + mes.content + '<|im_end|>\n'
        messages += '<|im_start|>assistant\n'
        return messages


class ModelInferenceGguf(ModelInference):
    streaming_args = {}

    def __init__(self, model_path, file_name, n_gpu_layers, n_batch, n_ctx):
        self.prompter = PromptManager()

        if model_path is not None and os.path.exists(model_path):
            download_path = model_path
        elif file_name is not None and os.path.exists(file_name):
            download_path = file_name
        else:
            download_path = hf_hub_download(repo_id=model_path, filename=file_name)

        llm = LlamaCpp(
            model_path=download_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            streaming=True,
            stop=['<|im_end|>', '</tool_call>'],
            max_tokens=1500,
            temperature=0.8,
            #repetition_penalty=1.1,
        )

        prompt = TemplateMessagesPrompt(
            input_variables=['history', ],
        )

        self.chain = prompt | llm

    async def run_inference(self, prompt, finish_tool_name):
        history = ChatMessageHistory()
        for mes in prompt:
            if mes['role'] == 'user':
                history.add_user_message(mes['content'])
            elif mes['role'] == 'assistant':
                history.add_ai_message(mes['content'])
            elif mes['role'] == 'system':
                history.add_message(SystemMessage(mes['content']))
            else:
                history.add_message(ToolMessage(mes['content'], tool_call_id=''))

        recieved_message = '<|im_start|>assistant\n'
        self.streaming_args = {}
        async for chunk in self.chain.astream({"history": history.messages}):
            recieved_message += chunk
            if '<tool_call>' in recieved_message:
                check_message = recieved_message.split('<tool_call>', 1)[1].replace('”', '"').replace("´", "'")
                if '{' in check_message:
                    check_message = '{' + check_message.split('{', 1)[1]
                    rsplit_size = check_message.count('}') - check_message.count('{') + 1
                    if rsplit_size > 0:
                        check_message = check_message.rsplit('}', rsplit_size)[0] + '}'
                    check_dict = force_parse_json(check_message)
                    if check_dict is not None and 'name' in check_dict and check_dict['name'] == finish_tool_name and 'arguments' in check_dict:
                        self.streaming_args = check_dict['arguments']
                    if check_message.count('{') <= check_message.count('}'):
                        break
        if recieved_message.count('<tool_call>') > recieved_message.count('</tool_call>'):
            recieved_message += '</tool_call>'
        recieved_message += '<|im_end|>'
        return recieved_message

    def process_completion_and_validate(self, completion, chat_template):

        assistant_message = get_assistant_message(completion, chat_template, '<|im_end|>')

        if assistant_message:
            validation, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)

            if validation:
                inference_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                return tool_calls, assistant_message, error_message
        else:
            inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")

    def get_streaming_args(self):
        return self.streaming_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recursive function calling loop")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to use json mode examples")
    parser.add_argument("--query", type=str, default="I need the current stock price of Tesla (TSLA)")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    args = parser.parse_args()

    # specify custom model path
    if args.model_path or args.file_name:
        inference = ModelInferenceGguf(args.model_path, args.file_name, 999, 128, 4096)
    else:
        model_path = 'NousResearch/Hermes-3-Llama-3.1-8B-GGUF'
        file_name = 'Hermes-3-Llama-3.1-8B.Q6_K.gguf'
        inference = ModelInferenceGguf(model_path, file_name, 999, 128, 4096)
        
    # Run the model evaluator
    inference.generate_function_call(args.query, args.chat_template, args.num_fewshot, args.max_depth)

    del inference