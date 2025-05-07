import time
import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
import requests
import json
from rag.src.pipeline import EmoLLMRAG

print("downloading model")
base_path = "model"
os.system(f"modelscope download --model haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit --local_dir {base_path}")
print("model downloaded")

print("loading model")

print("downloading libs")
os.system(f"pip install faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
print("libs downloaded")

print("loading model")

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=nf4_config, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.use_default_system_prompt = False
rag_obj = EmoLLMRAG(model)
print("model loaded")

prompt_style = """
### Instruction:
你是Care，一个心理咨询AI助手，基于deepseek-r1微调模型，能够用专业的心理知识回答来访者的问题。每次回答问题前，需要结合联网搜索结果：{}以及本地知识库内容：{}进行思考，并将思考过程放在<think>和</think>之间，如果联网搜索结果和本地知识库内容均为空，则自己思考，然后再根据思考进行回答，回答放在</think>之后。

### Question:
{}

### Response:
<think>
"""

LANGSEARCH_API_URL = "https://api.langsearch.com/v1/web-search"
LANGSEARCH_API_KEY = os.getenv('LANGSEARCH_API_KEY') 
def langsearch(query, max_results=5):
    payload = json.dumps({
    "query": query,
    "freshness": "noLimit",
    "summary": True,
    "count": 10
    })
    headers = {
    'Authorization': f'Bearer {LANGSEARCH_API_KEY}',
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", LANGSEARCH_API_URL, headers=headers, data=payload)
    if response.status_code == 200:
        print("Response Success: 200")
        results = json.loads(response.text).get("data").get("webPages").get("value")
        search_results = []
        for result in results:
            title = result.get("name", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            search_results.append(f"标题: {title}\n摘要: {snippet}\n链接: {url}\n")
        return "\n".join(search_results)
    else:
        print(f"Error: {response.status_code}")
        return ""

def format_time(seconds_float):
    total_seconds = int(round(seconds_float))
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

DESCRIPTION = '''
# 🧠 An AI assistant with extensive knowledge in psychology, and my name is Care.

## 🚀 Overview
This model is finetuned on deepseek-r1. If this repo helps you, star and share it ❤️. This repo will be continuously merged into EmoLLM.

## ✨ Functions
✅Provide an interactive chat interface for psychological consultation seekers.

✅Integrate knowledge retrieval 

✅Integrate web searching

❌customized tts

❌Virtual mental companion 

## ⚠️ issue status
- 2025.4.29 fix bug of clearing and stopping op.
- 2025.5.3 web search supports.
- 2025.5.5 rag supports. (demo code, needs to be checked)
- 2025.5.7 fix bug of rag.

## 🙏 Acknowledgments
We are grateful to Modelscope for supporting this project with resources.

The rag codes are based on [EmoLLM](https://github.com/SmartFlowAI/EmoLLM)

## 🤝 Contributing
Feel free to contribute to this project via our [github repo](https://github.com/HaiyangPeng/careyou). Grow together!
'''

CSS = """
.spinner {
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 8px;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.thinking-summary {
    cursor: pointer;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 4px;
    margin: 4px 0;
}
.thought-content {
    padding: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    margin: 5px 0;
}
.thinking-container {
    border-left: 3px solid #facc15;
    padding-left: 10px;
    margin: 8px 0;
    background: #ffffff;
}
details:not([open]) .thinking-container {
    border-left-color: #290c15;
}
details {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
    transition: border-color 0.2s;
}
"""

def user(message, history):
    return "", history + [[message, None]]

class ParserState:
    __slots__ = ['answer', 'thought', 'in_think', 'start_time', 'last_pos', 'total_think_time']
    def __init__(self):
        self.answer = ""
        self.thought = ""
        self.in_think = False
        self.start_time = 0
        self.last_pos = 0
        self.total_think_time = 0.0

def parse_response(text, state):
    buffer = text[state.last_pos:]
    state.last_pos = len(text)
    
    while buffer:
        if not state.in_think:
            think_start = buffer.find('<think>')
            if think_start != -1:
                state.answer += buffer[:think_start]
                state.in_think = True
                state.start_time = time.perf_counter()
                buffer = buffer[think_start + 7:]
            else:
                state.answer += buffer
                break
        else:
            think_end = buffer.find('</think>')
            if think_end != -1:
                state.thought += buffer[:think_end]
                # Calculate duration and accumulate
                duration = time.perf_counter() - state.start_time
                state.total_think_time += duration
                state.in_think = False
                buffer = buffer[think_end + 8:]
            else:
                state.thought += buffer
                break
    
    elapsed = time.perf_counter() - state.start_time if state.in_think else 0
    return state, elapsed

def format_response(state, elapsed):
    answer_part = state.answer.replace('<think>', '').replace('</think>', '')
    collapsible = []
    collapsed = "<details open>"

    if state.thought or state.in_think:
        if state.in_think:
            # Ongoing think: total time = accumulated + current elapsed
            total_elapsed = state.total_think_time + elapsed
            formatted_time = format_time(total_elapsed)
            status = f"❤️ Care的思考过程 {formatted_time}"
        else:
            # Finished: show total accumulated time
            formatted_time = format_time(state.total_think_time)
            status = f"🐷 Care的回答 {formatted_time}"
            collapsed = "<details>"
        collapsible.append(
            f"{collapsed}<summary>{status}</summary>\n\n<div class='thinking-container'>\n{state.thought}\n</div>\n</details>"
        )

    return collapsible, answer_part

def generate_response(history, temperature, top_p, max_tokens, active_gen):
    user_message = history[-1][0]
    search_results = langsearch(user_message)
    if search_results:
        print("联网搜索结果：", search_results)
    else:
        print("未搜索到准确信息，将按照原始流程进行推理")

    conversation = []
    for user, assistant in history[:-1]:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    retrieval_content = rag_obj.get_retrieval_content(user_message)
    retrieval_content = " ".join(retrieval_content)
    if retrieval_content:
        print("知识库搜索结果：", retrieval_content)
    else:
        print("未搜索到准确信息，将按照原始流程进行推理")

    conversation.append({"role": "user", "content": prompt_style.format(search_results, retrieval_content, user_message)})

    input_ids = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    print("input_ids1: ", input_ids)
    input_ids = tokenizer([input_ids], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids.input_ids,
        streamer=streamer,
        do_sample=True,
        max_new_tokens=max_tokens,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    full_response = "<think>"
    state = ParserState()
    last_update = 0

    try:
        for chunk in streamer:
            if not active_gen[0]:
                break
            print(chunk, end="", flush=True)
            if chunk:
                full_response += chunk
                state, elapsed = parse_response(full_response, state)

                collapsible, answer_part = format_response(state, elapsed)
                history[-1][1] = "\n\n".join(collapsible + [answer_part])
                yield history

        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history[-1][1] = "\n\n".join(collapsible + [answer_part])
        yield history

    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield history
    finally:
        active_gen[0] = False
        t.join()

with gr.Blocks(css=CSS) as demo:
    gr.Markdown(DESCRIPTION)
    active_gen = gr.State([False])
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=500,
        show_label=False,
        render_markdown=True
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Message",
            placeholder="Type your message...",
            container=False,
            scale=4
        )
        submit_btn = gr.Button("Send", variant='primary', scale=1)
    
    with gr.Column(scale=2):
        with gr.Row():
            clear_btn = gr.Button("Clear", variant='secondary')
            stop_btn = gr.Button("Stop", variant='stop')
        
        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p")
            max_tokens = gr.Slider(minimum=2048, maximum=32768, value=4096, step=64, label="Max Tokens")

    gr.Examples(
        examples=[
            ["你是谁呀"],
            ["我很难过，爸妈不爱我"],
            ["爸妈老是说我笨"]
        ],
        inputs=msg,
        label="咨询例子"
    )
    
    submit_event = submit_btn.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        lambda: [True], outputs=active_gen
    ).then(
        generate_response, [chatbot, temperature, top_p, max_tokens, active_gen], chatbot
    )
    
    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        lambda: [True], outputs=active_gen
    ).then(
        generate_response, [chatbot, temperature, top_p, max_tokens, active_gen], chatbot
    )
    
    stop_btn.click(
        lambda: [False], None, active_gen, cancels=[submit_event]
    )
    
    clear_btn.click(
        lambda: [False], None, active_gen, cancels=[submit_event]
    ).then(
        lambda: None, None, chatbot, queue=False
    )

if __name__ == "__main__":
    demo.queue(api_open=False, max_size=20, default_concurrency_limit=20).launch(server_name="0.0.0.0", server_port=7860, max_threads=40)
