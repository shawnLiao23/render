from transformers import Pix2StructForConditionalGeneration
import shutil
import os

# 清除 Hugging Face 缓存
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# 重新下载模型（禁用 safetensors 作为备选方案）
deplot_model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/deplot",
    use_safetensors=False,  # 尝试不使用 safetensors
    local_files_only=False,  # 强制重新下载
)