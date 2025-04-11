import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from functools import lru_cache
import asyncio
import re
import requests
import hashlib
import random

# 模型加载
class AIGCGenerator:
    def __init__(self):
        # 加载Qwen语言模型的Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True,
            use_fast=False
        )
        # 加载Qwen语言模型
        self.text_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        # 设置chat模板，格式化输入用
        self.tokenizer.chat_template = """
        {% for message in messages %}
        {% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>
        {% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>
        {% else %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>
        {% endif %}
        {% endfor %}
        <|im_start|>assistant\n"""

        # 加载Stable Diffusion图像生成模型
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        self._optimize_pipeline()

    def _optimize_pipeline(self):
        # 图像生成内存优化
        self.sd_pipe.enable_attention_slicing()
        self.sd_pipe.enable_model_cpu_offload()
        self.sd_pipe.enable_xformers_memory_efficient_attention()
        torch.cuda.empty_cache()



# 生成模块
class ContentGenerator(AIGCGenerator):
    # 各种风格的Prompt模板配置
    PROMPT_CONFIG = {
        "小红书风格": {
            "system": """你是一位专业的美妆博主，请按以下结构生成原创文案（禁止重复示例）：
🔥【痛点标题】 (1个emoji+4-6字)
👉 问题场景：描述具体使用场景和问题
✅ 解决方案：用❗️标记3个核心卖点
🌟 使用效果：使用前后对比
🏷️ 标签：3-5个相关标签""",
            "example_input": "遮瑕膏",
            "example_output": """
🔥 熬夜救星！
👉 每天加班到凌晨，黑眼圈重到粉底都盖不住...
✅ 三大优势：
❗️ 三色分区遮瑕
❗️ 水润不卡粉
❗️ 持妆12小时
🌟 薄涂一层就能伪装好气色
🏷️ #遮瑕推荐 #美妆神器 #打工人必备"""
        },

        "ins风格": {
        "system": """你是有百万粉丝的INS博主，请按以下结构使用轻松风格生成原创文案（禁止重复示例）：
🌿【LIFE HACK】 (1个氛围感emoji+3-5英文词)
☕ 痛点场景：用生活化场景制造共鸣
🛋️ 解决方案：用🪐标记3个设计亮点
📸 视觉对比：before-after的意境描述
🌐 标签组合：2个#Lifestyle + 1个#品牌调性""",
        "example_input": "香薰机",
        "example_output": """
🌿 Urban Oasis
☕ 加班回家总闻到外卖盒酸味
🛋️ 空间改造术：
🪐 雾化粒子细至0.3μm
🪐 月光波纹光影
🪐 语音调节浓度
📸 从"合租屋异味"到"私人SPA馆"
🌐 #HomeDecor #WellnessLiving #MUJIvibes"""
        },

        "B站风格": {
        "system": """你是会整活的B站UP主，请按以下结构生成有趣原创文案（禁止重复示例）：
        🐶【离谱痛点】 (1个魔性emoji+口语化短句)
        🤯 崩溃实录：夸张化演绎真实窘境
        🚨 真香警告：用⚡标3个逆天功能
        🌈 效果暴击：对比段子+热梗call back
        🎮 必带梗：#电子宠物 + #贫民窟XX""",
        "example_input": "电动牙刷",
        "example_output": """
        🐶 刷牙刷出火星子！
        🤯 "这震动怕不是装了小马达"（弹幕：电钻警告！）
        🚨 黑科技三连：
        ⚡ 压力感应自动降频
        ⚡ 舌苔清洁模式
        ⚡ 续航吊打某果
        🌈 从"牙龈刺客"到"口腔大保健"（弹幕：牙医连夜改行）
        🎮 #打工人生存包 #宿舍违禁品"""
        }
      }

    def __init__(self):
        super().__init__()
        # 百度翻译API的配置
        self.baidu_appid = "20250410002329355"  #id
        self.baidu_key = "p5GCT04rp8Il7mr62W6n" #密钥

    @lru_cache(maxsize=50)
    def generate_text(self, keywords: str, style: str) -> str:
        # 获取所选风格配置
        config = self.PROMPT_CONFIG[style]

        # 构建输入对话消息序列
        messages = [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": f"示例产品：{config['example_input']}"},
            {"role": "assistant", "content": config["example_output"]},
            {"role": "user", "content": f"新产品：{keywords}"},
            {"role": "user", "content": "请生成全新内容，不要使用示例中的任何具体描述"}
        ]

        # 转为模型需要的Prompt格式
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        #print("最终用于生成的 Prompt：", formatted_prompt)

        # 编码并生成内容
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        outputs = self.text_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.5,
            do_sample=True,
            num_beams=3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("原始输出 >>>", full_text)

        return self._process_output(full_text, config["example_output"])


    def _process_output(self, text: str, example: str) -> str:
        # 清洗模型生成的原始文本，提取有效内容
        split_markers = ["请生成全新内容",f"新产品：","<|im_start|>assistant"]

        for marker in split_markers:
            if marker in text:
                new_content = text.split(marker)[-1]
                break
        else:
            new_content = text

        # 清理系统标记和示例内容
        clean_text = re.sub(r"<\|.*?\|>|user|assistant|system", "", new_content)
        clean_text = clean_text.replace(example.split("👉")[0], "").strip()

        lines = clean_text.split("\n")
        formatted_lines = []
        section_found = False

        # 匹配行首 emoji 字符：包括表情符号区、部分 Unicode 特符号区
        emoji_pattern = re.compile(r"^[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测以 emoji 开头的段落，换行美化显示
            if emoji_pattern.match(line):
                formatted_lines.append(f"\n{line}")
                section_found = True
            elif section_found:
                formatted_lines.append(line)

        return "\n".join(formatted_lines).strip()[:1000]

    async def generate_image(self, keywords: str, style: str):
        # 构造中文prompt用于图像
        prompt_cn = f"{keywords} 产品照片，{style}风格，高质量，高清"
        prompt_en = self._translate_to_en(prompt_cn)

        # 图像生成
        return self.sd_pipe(
            prompt=prompt_en,
            negative_prompt="text, watermark, low quality",
            width=768,
            height=512,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

    def _translate_to_en(self, text):
        # 调用百度翻译接口
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((self.baidu_appid + text + salt + self.baidu_key).encode()).hexdigest()
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        params = {
            "q": text,
            "from": "zh",
            "to": "en",
            "appid": self.baidu_appid,
            "salt": salt,
            "sign": sign
        }
        try:
            res = requests.get(url, params=params, timeout=5)
            result = res.json()
            if "trans_result" in result:
                return result["trans_result"][0]["dst"]
            else:
                print("翻译失败：", result)
                return "high quality product photo"
        except Exception as e:
            print("翻译接口异常：", e)
            return "high quality product photo"


# Gradio 界面构建类
class AIGCApp:
    def __init__(self):
        self.generator = ContentGenerator()

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft(), css=".output-text {font-size: 14px}") as demo:
            gr.Markdown("# 🛍️ 电商文案生成器 ")

            with gr.Row():
                with gr.Column():
                    keywords = gr.Textbox(label="产品名称", placeholder="输入商品关键词（如：珍珠项链）")
                    style_selector = gr.Dropdown(
                        list(ContentGenerator.PROMPT_CONFIG.keys()),
                        label="内容风格",
                        value="小红书风格"
                    )
                    generate_btn = gr.Button("立即生成", variant="primary")
                    gr.Examples(examples=[["气垫粉扑"], ["头戴耳机"]], inputs=[keywords])

                with gr.Column():
                    text_output = gr.Textbox(label="生成文案", lines=10, elem_classes=["output-text"])
                    image_output = gr.Image(label="产品配图", height=400)

            generate_btn.click(self._disable_btn, None, generate_btn).then(
                self._generate_content,
                inputs=[keywords, style_selector],
                outputs=[text_output, image_output]
            ).then(self._enable_btn, None, generate_btn)

            return demo

    async def _generate_content(self, keywords, style):
        # 并行生成文案和图片
        text = await asyncio.to_thread(self.generator.generate_text, keywords, style)
        image = await self.generator.generate_image(keywords, style)
        return text, image

    def _disable_btn(self):
        return gr.update(interactive=False)

    def _enable_btn(self):
        return gr.update(interactive=True)





if __name__ == "__main__":
    app = AIGCApp().create_interface()
    app.queue().launch(
        debug=True,
        share=True
    )