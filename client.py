#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import gradio as gr

def main_ui():
    global mcp_params_json, mcp_params_base

    chat_settings = None
    settings_file_path = 'settings/chat_settings.json'
    if os.path.isfile(settings_file_path):
        with open(settings_file_path, 'r') as f:
            chat_settings = json.load(f)
    else:
        chat_settings = {}

    file_path = 'settings/mcp_config.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            mcp_params_json = f.read()
    else:
        mcp_params_json = r'{}'
    mcp_params_base = json.loads(mcp_params_json)

    with gr.Blocks(analytics_enabled=False) as runner_interface:
        with gr.Row():
            gr.Markdown(value='## Chat')
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                text_input = gr.Textbox(lines=2, label='')
                with gr.Row():
                    btn_generate = gr.Button(value='Chat', variant='primary')
                with gr.Row():
                    btn_regenerate = gr.Button(value='Regenerate')
                    btn_continue = gr.Button(value='Continue', interactive=False)
                    btn_remove_last = gr.Button(value='Remove last')
                    btn_clear = gr.Button(value='Clear all')
                    btn_abort = gr.Button(value='Abort', interactive=False)
                with gr.Row():
                    txt_file_path = gr.Dropdown(value='', allow_custom_value=True, label='File name or path')
                    btn_load = gr.Button(value='Load')
                    #btn_load.click(fn=chatgpt_load, inputs=[txt_file_path, chatbot], outputs=chatbot)
                    btn_save = gr.Button(value='Save')
                    #btn_save.click(fn=chatgpt_save, inputs=[txt_file_path, chatbot], outputs=txt_file_path)
        with gr.Row():
            gr.Markdown(value='## Settings')
        with gr.Row():
            llama_cpp_model_file = gr.Textbox(label='Model File Path (*.gguf)')
        with gr.Row():
            with gr.Column():
                llama_cpp_n_gpu_layers = gr.Number(label='n_gpu_layers')
            with gr.Column():
                llama_cpp_n_batch = gr.Number(label='n_batch')
            with gr.Column():
                llama_cpp_n_ctx = gr.Number(label='n_ctx')
        with gr.Row():
            #with gr.Column():
            #    llama_cpp_system_message_language = gr.Dropdown(value='English', allow_custom_value=False, label='System Message Language', choices=['English', 'Japanese'])
            with gr.Column():
                btn_llama_cpp_save = gr.Button(value='Save And Reflect', variant='primary')
        def llama_cpp_save(path: str, n_gpu_layers: int, n_batch: int, n_ctx: int,
            #system_message_language: str
        ):
            chat_settings['llama_cpp_model'] = path
            chat_settings['llama_cpp_n_gpu_layers'] = n_gpu_layers
            chat_settings['llama_cpp_n_batch'] = n_batch
            chat_settings['llama_cpp_n_ctx'] = n_ctx
            #chat_settings['llama_cpp_system_message_language'] = system_message_language
            os.makedirs('settings', exist_ok=True)
            with open('settings/chat_settings.json', 'w') as f:
                json.dump(chat_settings, f)
            #chat_gpt_api.load_settings(**chat_settings)
        btn_llama_cpp_save.click(fn=llama_cpp_save, inputs=[llama_cpp_model_file, llama_cpp_n_gpu_layers, llama_cpp_n_batch, llama_cpp_n_ctx,
            #llama_cpp_system_message_language
        ])
        with gr.Row():
            txt_json_settings = gr.Textbox(value='', label='MCP Servers')
        with gr.Row():
            with gr.Column():
                btn_settings_save = gr.Button(value='Save', variant='primary')
                def json_save(settings: str):
                    os.makedirs('settings', exist_ok=True)
                    with open('settings/mcp_config.json', 'w') as f:
                        f.write(settings)
                btn_settings_save.click(fn=json_save, inputs=txt_json_settings)
            with gr.Column():
                btn_settings_reflect = gr.Button(value='Reflect settings', variant='primary')
                def json_reflect(settings: str):    
                    global mcp_params_json, mcp_params_base
                    mcp_params_json = settings
                    mcp_params_base = json.loads(mcp_params_json)
                btn_settings_reflect.click(fn=json_reflect, inputs=txt_json_settings)
        
        set_interactive_items = [text_input, btn_generate, btn_regenerate,
            btn_remove_last, btn_clear, btn_load, btn_save,
            llama_cpp_model_file, llama_cpp_n_gpu_layers, llama_cpp_n_batch, btn_llama_cpp_save,
            llama_cpp_n_ctx,
            txt_json_settings, btn_settings_save, btn_settings_reflect]

        def on_load():
            lines = mcp_params_json.count('\n') + 2
            json_settings = gr.update(lines=lines, max_lines=lines + 5, value=mcp_params_json)

            if not 'llama_cpp_n_gpu_layers' in chat_settings:
                chat_settings['llama_cpp_n_gpu_layers'] = 20
            if not 'llama_cpp_n_batch' in chat_settings:
                chat_settings['llama_cpp_n_batch'] = 128
            if not 'llama_cpp_n_ctx' in chat_settings:
                chat_settings['llama_cpp_n_ctx'] = 4096
            #if not 'llama_cpp_system_message_language' in chat_settings:
            #    chat_settings['llama_cpp_system_message_language'] = 'English'

            ret = [json_settings, chat_settings['llama_cpp_n_gpu_layers'], chat_settings['llama_cpp_n_batch'], chat_settings['llama_cpp_n_ctx']]

            for key in ['llama_cpp_model', 
                #'llama_cpp_system_message_language'
            ]:
                if key in chat_settings:
                    ret.append(chat_settings[key])
                else:
                    ret.append('')
            
            return ret

        runner_interface.load(on_load, outputs=[
            txt_json_settings,
            llama_cpp_n_gpu_layers, llama_cpp_n_batch, llama_cpp_n_ctx,
            llama_cpp_model_file, 
            #llama_cpp_system_message_language,
        ])

    return runner_interface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_browser_open', action='store_true')
    args = parser.parse_args()

    block_interface = main_ui()
    block_interface.queue()
    block_interface.launch(inbrowser=(not args.disable_browser_open))