import argparse
import torch
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import functions
from prompter import PromptManager
from validator import validate_function_call_schema

from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)

from langchain_core.utils.function_calling import convert_to_openai_tool

class ModelInference:
    def __init__(self, model_path, chat_template, load_in_4bit):
        inference_logger.info(print_nous_text_art())
        self.prompter = PromptManager()
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = get_chat_template(chat_template)
        
        inference_logger.info(self.model.config)
        inference_logger.info(self.model.generation_config)
        inference_logger.info(self.tokenizer.special_tokens_map)

    def process_completion_and_validate(self, completion, chat_template):

        assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

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
        
    async def execute_function_call(self, tool_call, langchain_tools):
        function_name = tool_call.get("name")
        function_args = tool_call.get("arguments", {})
        if langchain_tools is None:
            function_to_call = getattr(functions, function_name, None)
            inference_logger.info(f"Invoking function call {function_name} ...")
            function_response = function_to_call(*function_args.values())
        else:
            for loop_tool in langchain_tools:
                if loop_tool.name == function_name:
                    function_to_call = loop_tool
            inference_logger.info(f"Invoking function call {function_name} ...")
            function_response = await function_to_call.ainvoke(function_args)

        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict
    
    async def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)
        return completion

    def generate_function_call(self, query, chat_template, num_fewshot, max_depth=5, tools=None, history=[]):
        asyncio.run(self.generate_function_call_async(query, chat_template, num_fewshot, max_depth, tools, history))

    async def generate_function_call_async(self, query, chat_template, num_fewshot, max_depth=5, tools=None, history=[], finish_tool_name=None):
        try:
            depth = 0
            user_message = query #f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
            chat = history + [{"role": "user", "content": user_message}]
            if tools is None:
                langchain_tools = None
                tools = functions.get_openai_tools()
            else:
                langchain_tools = tools
                tools = [convert_to_openai_tool(f) for f in tools]
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            input_prompt_count = len(prompt)
            completion = await self.run_inference(prompt, finish_tool_name)

            async def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                tool_calls, assistant_message, error_message = self.process_completion_and_validate(completion, chat_template)
                prompt.append({"role": "assistant", "content": assistant_message})

                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                if tool_calls:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

                    for tool_call in tool_calls:
                        validation, message = validate_function_call_schema(tool_call, tools)
                        if validation:
                            try:
                                function_response = await self.execute_function_call(tool_call, langchain_tools)
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                inference_logger.info(f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}")
                            except Exception as e:
                                inference_logger.info(f"Could not execute function: {e}")
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})

                    for tool_call in tool_calls:
                        if finish_tool_name == tool_call.get("name"):
                            return prompt[input_prompt_count:]

                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return None

                    completion = await self.run_inference(prompt, finish_tool_name)
                    return await recursive_loop(prompt, completion, depth)
                elif error_message:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return None

                    completion = await self.run_inference(prompt, finish_tool_name)
                    return await recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    if finish_tool_name is not None:
                        depth += 1
                        if depth >= max_depth:
                            print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                            return None
                        prompt.append({"role": "user", "content": f"Please execute \"{finish_tool_name}\" tool with this contents."})
                        completion = await self.run_inference(prompt, finish_tool_name)
                        return await recursive_loop(prompt, completion, depth)
                    return prompt[input_prompt_count:] #assistant_message

            return await recursive_loop(prompt, completion, depth)

        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recursive function calling loop")
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to use json mode examples")
    parser.add_argument("--load_in_4bit", type=str, default="False", help="Option to load in 4bit with bitsandbytes")
    parser.add_argument("--query", type=str, default="I need the current stock price of Tesla (TSLA)")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    args = parser.parse_args()

    # specify custom model path
    if args.model_path:
        inference = ModelInference(args.model_path, args.chat_template, args.load_in_4bit)
    else:
        model_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
        inference = ModelInference(model_path, args.chat_template, args.load_in_4bit)
        
    # Run the model evaluator
    inference.generate_function_call(args.query, args.chat_template, args.num_fewshot, args.max_depth)
