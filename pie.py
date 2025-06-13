import os
from io import BytesIO
import io
import streamlit as st
import openai
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import base64
from typing import Dict, Optional
from PIL import Image

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from transformers import Pix2StructForConditionalGeneration


# 加载 DePlot 模型（只加载一次）
deplot_processor = Pix2StructProcessor.from_pretrained('google/deplot')
deplot_model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-3d63de1f971640958d6cbbe98ae670b7"  # 替换为你的实际API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # 确认最新API地址

def extract_table_from_image_deplot(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = deplot_processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    predictions = deplot_model.generate(**inputs, max_new_tokens=512)
    return deplot_processor.decode(predictions[0], skip_special_tokens=True)

class GraphGenerator:
    def __init__(self):
        self.safe_modules = {
            "plt": plt,
            "np": np,
            "math": __import__("math")
        }
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.data_save_folder = "generated_data_pie"
        if not os.path.exists(self.data_save_folder):
            os.makedirs(self.data_save_folder)

        self.counter_file_path = os.path.join(self.data_save_folder, "counter.txt")
        if os.path.exists(self.counter_file_path):
            with open(self.counter_file_path, encoding='utf-8') as f:
                self.data_counter = int(f.read())
        else:
            self.data_counter = 1
    def extract_color_palette(self, image_path: str, max_colors: int = 10) -> list:
        """
        从输入图像提取主要颜色调色板，基于K-means聚类，忽略亮度过高的颜色
        """

        def rgb_to_hsv(rgb):
            # 将RGB转为HSV色彩空间
            rgb = np.array(rgb) / 255.0
            maxc = np.max(rgb)
            minc = np.min(rgb)
            v = maxc
            if minc == maxc:
                return (0.0, 0.0, v)
            s = (maxc - minc) / maxc
            rc = (maxc - rgb[0]) / (maxc - minc)
            gc = (maxc - rgb[1]) / (maxc - minc)
            bc = (maxc - rgb[2]) / (maxc - minc)
            h = 0.0
            if rgb[0] == maxc:
                h = bc - gc
            elif rgb[1] == maxc:
                h = 2.0 + rc - bc
            else:
                h = 4.0 + gc - rc
            h = (h / 6.0) % 1.0
            return (h, s, v)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.width * img.height > 300000:
                img = img.resize((300, 300))  # 保持更多细节

            data = np.array(img)
            pixels = data.reshape(-1, 3)

            # 多维度过滤条件
            filtered_pixels = []
            for pixel in pixels:
                r, g, b = [int(x) for x in pixel]
                h, s, v = rgb_to_hsv(pixel)

                cond1 = v < 0.9  # 排除过亮颜色
                cond2 = s > 0.2  # 保证饱和度
                cond3 = not (abs(r - g) < 30 and abs(g - b) < 30)  # 排除近似灰色

                if cond1 and cond2 and cond3:
                    filtered_pixels.append(pixel)

            if not filtered_pixels:
                return ["#FF9999", "#66B3FF"]  # 默认备用颜色

            pixels = np.array(filtered_pixels, dtype=np.float32)

            # 动态调整聚类数量
            actual_colors = min(max_colors, len(np.unique(pixels, axis=0)))
            if actual_colors < 2:
                return ['#%02x%02x%02x' % tuple(pixels[0])]

            # K-means聚类改进版
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=actual_colors,
                random_state=42,
                batch_size=1024,
                n_init=3
            )
            kmeans.fit(pixels)

            # 按聚类大小排序，优先选择主要颜色
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            sorted_colors = kmeans.cluster_centers_[np.argsort(-counts)]

            # 转换为十六进制
            palette = []
            for color in sorted_colors:
                hex_color = '#%02x%02x%02x' % tuple(color.astype(int))

                # 确保颜色差异度 (与已选颜色对比)
                if not palette or all(
                        self._color_distance(hex_color, c) > 30
                        for c in palette
                ):
                    palette.append(hex_color)
                    if len(palette) >= max_colors:
                        break

            return palette

    @staticmethod
    def _color_distance(c1: str, c2: str) -> float:
        """计算两个颜色之间的欧氏距离"""

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

    def call_gpt_and_generate(self, initial_instruction: str, requirement: str, student_answer: str, image_path: Optional[str] = None, model: str = "deepseek-chat", output_format: str = "json") -> Dict:
        try:
            system_content = """You are an IELTS Academic Writing Task 1 expert. Your tasks:
            1. Analyze student answers to extract all numerical data without omissions. 
            2. Use a table extracted from the image (via DePlot, in JSON format) **only as a secondary reference**. Follow these strict rules:
               - For each category in the DePlot table:
                 - If the category appears in the student answer **and** includes a specific number:
                     - If this number **matches** DePlot: include it using that number.
                     - If this number **differs** from DePlot: **use the student's number only** — never override it with DePlot.
                 - If the category appears in the student answer **but without a number**, and only a relative comparison is made (e.g., "higher than X"):
                     - Use DePlot to check if the description is correct:
                         - If correct: include the category using the DePlot value.
                         - If incorrect: include the category in the output using a **reasonable estimated value** that matches the trend described — do **not** copy DePlot's value directly.
                 - If a category is **not mentioned at all** in the student answer: do not include it in the output.

            3. You must **never override** student-provided numerical values with DePlot's values.
            4. Only use DePlot values to fill in cases with comparisons **when no explicit number is provided**.
            5. Generate structured data representations of charts. Keep the chart title concise and under 10 words.
            6. Output must be in safe JSON format.
            7. Generate structured data representations with original color palette
            8. Preserve exact color hex codes from reference image

            Output format:
            - JSON structure:
              {"title":"...", "categories":[], "x_label":"...", "y_label":"...", 
               "series":[{"label":"...", "values":[]}], "chart_type":"pie", 
               "style":{"orientation":"...","color_palette": [str]}}
            - Only include categories explicitly mentioned in the student answer.
            - Do not infer, hallucinate, or guess extra categories.
            - Do not include harmful or invalid code.
            - If the chart_type is 'pie', generate only one series, and use 'categories' and 'values' directly as pie sections.
            - Do not infer or hallucinate categories. Only include categories that are explicitly mentioned by the student.
            - Never include harmful code
            - Prioritize completeness and faithfulness to the student answer over visual similarity to any reference image.
            """

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Task Context: {initial_instruction}"},
                {"role": "user", "content": f"Official Requirement: {requirement}"},
                {"role": "user", "content": f"""
                    ## Student Answer Analysis Task
                    * Primary Data Source: STUDENT ANSWER
                    * Conflict Resolution: Prioritize student's version when conflicting with sample
                    * ALL mentioned categories in the student's answer MUST be present in the output JSON, even if not present in the reference image.
                    [BEGIN STUDENT ANSWER]
                    {student_answer}
                    [END STUDENT ANSWER]
                    """}
            ]

            if image_path:
                # 提取并添加颜色调色板
                palette = self.extract_color_palette(image_path)
                deplot_output = extract_table_from_image_deplot(image_path)
                messages.append({
                    "role": "user",
                    "content": f"The following table was extracted from the provided chart image using DePlot:\n\n{deplot_output}"
                })

                if palette:
                    messages.append({
                        "role": "user",
                        "content": f"COLOR CONSTRAINTS: Must use these exact colors {palette} in chart"
                    })
                # # 添加编码后的图像
                # messages.append(self._encode_image(image_path))


            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
            )
            print(response.choices[0].message.content)

            return self._process_response(response, output_format,image_path)

        except Exception as e:
            print(f"API Error: {str(e)}")
            return {"error": str(e)}
    def _encode_image(self, image_path: str) -> Dict:
        """
        将图像压缩到原尺寸的 75%，然后转为 base64 以纯文本形式传递
        """

        # 打开并压缩图像
        img = Image.open(image_path)
        width, height = img.size
        new_size = (int(width * 0.5), int(height * 0.5))
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 保存压缩图像到内存 buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format="PNG", optimize=True, quality=95)

        # 编码为 base64
        encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content_str = f"Reference image provided: data:image/png;base64,{encoded_data}"

        return {
            "role": "user",
            "content": content_str
        }


    def infer_orientation_from_image(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            return "horizontal" if width > height else "vertical"

    def _process_response(self, response, output_format: str, image_path) -> Dict:
        """
        响应处理方法
        """
        try:
            content = response.choices[0].message.content
            # 调试：将完整返回内容写入 debug.txt 便于查看
            with open("debug.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("=== Raw Response Content 已保存到 debug.txt ===")

            if output_format == "json":
                return self._parse_json(content, image_path)
            return self._safe_execute_code(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        except Exception as e:
            return {"error": str(e)}

    def _parse_json(self, content: str, image_path) -> Dict:
        """
        JSON解析方法，自动提取 markdown 格式中的 JSON 部分
        """
        # 尝试提取第一个 ```json ... ``` 包裹的 JSON 内容
        json_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
        match = json_pattern.search(content)
        if match:
            json_str = match.group(1)
        else:
            # 如果没有找到 markdown 包裹，则尝试匹配最外层的 { ... }
            json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
            match = json_pattern.search(content)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError("No valid JSON block found in response.")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extracted JSON is invalid: {e}")

        if not {"categories", "series"}.issubset(data.keys()):
            raise ValueError("Missing required keys in JSON data")

        # 补足 pie chart 百分比，不足则添加 'missing'
        total = sum(data["series"][0]["values"])
        if total < 100:
            missing_value = round(100 - total, 1)
            data["categories"].append("Missing")
            data["series"][0]["values"].append(missing_value)
            data.setdefault("style", {}).setdefault("missing_index", len(data["categories"]) - 1)

        self._plot_from_json(data, image_path)
        return data

    def _increment_counter(self):
        """递增计数器并保存到文件中"""
        self.data_counter += 1
        with open(self.counter_file_path, "w") as f:
            f.write(str(self.data_counter))

    def _plot_from_json(self, data: Dict, image_path):
        """
        从 JSON 生成柱状图（可横向或纵向）
        """
        chart_type = data.get("chart_type", "bar")
        categories = data["categories"]
        num_categories = len(categories)
        y = np.arange(num_categories)
        series = data["series"]
        colors = data.get("style", {}).get("color_palette", ["black", "gray"])

        # IF pie chart
        if chart_type == "pie":
            self._plot_pie_chart(data)
            return

        # Orientation of bar
        orientation = self.infer_orientation_from_image(image_path)
        data["style"]["orientation"] = orientation
        # Height of bar
        height = 0.35  # 条形高度

        plt.figure(figsize=(10, 6))

        for idx, series in enumerate(series):
            values = series["values"]
            label = series["label"]
            color = colors[idx % len(colors)]

            offset = (idx - (len(series) - 1) / 2) * height

            if orientation == "horizontal":
                plt.barh(y + offset, values, height, label=label, color=color, alpha=0.7)
            else:  # vertical
                plt.bar(y + offset, values, height, label=label, color=color, alpha=0.7)

        # 设置标签和标题
        if orientation == "horizontal":
            plt.xlabel(data["y_label"])
            plt.ylabel(data["x_label"])
            plt.yticks(y, categories)
        else:
            plt.ylabel(data["y_label"])
            plt.xlabel(data["x_label"])
            plt.xticks(y, categories, rotation=30)

        plt.title(data["title"])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_pie_chart(self, data: Dict):
        categories = data["categories"]
        values = data["series"][0]["values"]
        colors = data.get("style", {}).get("color_palette", None)
        title = data.get("title", "")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制扇形
        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,
            radius=1.0,
            wedgeprops=dict(width=1.0, edgecolor='white')
        )

        # 若存在 Missing 类，绘制斜线
        missing_index = data.get("style", {}).get("missing_index")
        if missing_index is not None and 0 <= missing_index < len(wedges):
            wedges[missing_index].set_facecolor((1.0, 0.3, 0.3, 0.4))  # 红色背景 + 半透明
            wedges[missing_index].set_hatch('//')  # 斜线填充
            wedges[missing_index].set_edgecolor('black')  # 加边框

        # 计算角度，放置 label
        angles = [(wedge.theta2 + wedge.theta1) / 2.0 for wedge in wedges]

        for i, angle in enumerate(angles):
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))

            x_text = 1.2 * x
            y_text = 1.2 * y

            label = f"{categories[i]}\n{values[i]}%"

            ha = 'left' if x >= 0 else 'right'
            ax.text(x_text, y_text, label, ha=ha, va='center', fontsize=10)

        ax.axis('equal')
        plt.title(title, fontsize=12, loc='center', y=1.08)

        plt.tight_layout()

        # 保存图像，并使用递增后的计数器作为文件名
        data_path = os.path.join(self.data_save_folder, f"answer{self.data_counter}.png")
        plt.savefig(data_path)

        # 在图形绘制完成后递增计数器
        self._increment_counter()

        # 显示图像
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption="Student Graph", use_container_width=True)

        # 输出图像的路径
        st.write(f"Data saved to: {os.path.abspath(self.data_save_folder)}")

    def _safe_execute_code(self, content: str) -> Dict:
        """
        安全执行代码方法
        """
        code_block = self._extract_code(content)
        if not code_block:
            return {"error": "No valid code block"}
        try:
            restricted_globals = self.safe_modules.copy()
            local_env = {}
            exec(code_block, restricted_globals, local_env)
            plt.show()
            return {"status": "success", "output": local_env.get("figure")}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

    @staticmethod
    def _extract_code(content: str) -> Optional[str]:
        """
        提取代码块方法
        """
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else None


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 输入参数配置
    initial_instruction = (
        "Now I'll send you the Requirement, graph and Sample answer of the first Writing question of IELTS Academic. "
        "You need to learn how to reverse generate the graph according to the requirement and given answer which "
        "describes the graph by the materials I give to you."
    )

    # requirement = (
    #     """The pie chart below shows the proportion of different categories of families living in poverty in UK in 2002.
    #     Summarise the information by selecting and reporting the main features, and make comparisons where relevant.
    #     Write at least 150 words."""
    # )
    requirement = (
        """The pie chart gives information on UAE government spending in 2000. The total budget was AED 315 billion.
        Summarize the information by selecting and reporting the main features, and make comparisons where relevant.
        Write at least 150 words."""
    )

    # sample_answer = (
    #     "The provided bar chart shows the comparison between the numbers of male and female students enrolling in the "
    #     "research study in six different subjects, like linguistic, psychology, natural science, engineering programming "
    #     "and mathematics at an American University. Overall, the graph shows that there are more male students enrolled in "
    #     "the research field in comparison to female students. As per the provided illustration, both female and male students got "
    #     "an equal number of entries in natural science. As far as mathematics is concerned, male students had greater interest than "
    #     "females. Moreover, male entrants can be seen in all of the subjects except for linguistics. It is clear from the provided data "
    #     "that natural science turned to be the most sought subject for both genders as it recorded 400 entrants altogether (200 on each). "
    #     "In mathematics, men recorded another 200 entries as opposed to merely 50 female students. Additionally, in the psychology subject, "
    #     "there were almost 375 entrants. Even here, male students dominated at 200 and females were at 175. On the other hand, linguistic "
    #     "defined a completely different story as female enrollers toppled the number of male entrants at approximately 120 to 80, respectively."
    # )

    # student_answer = (
    #     """
    #     The pie chart inspects the different family types living in poor conditions in the UK in 2002.
    #     At a glance, in the given year, 16% of the entire households in the country were in circumstances of poverty. In comparison to the couples, singles struggled more. Talking about people with children, single parents presented the maximum percentage of 26% amongst all the specified categories, whereas couples with children reported a comparatively lesser percentage of 15%.
    #     As far as the people with no children are concerned, single people were of the similar percentage as couples with children, almost the same number for single people with children.  Coming to aged people, singles had a somewhat higher percentage in comparison to couples. Only 7% and 5% of the aged population had difficulties in their living conditions.
    #     """
    # )
    student_answer = (
            """
            The graph communicates the budget created by the UAE government in the year 2000. All in all, the essential targets that the government had were social security, health and education.

The largest space is covered by social security, such as pensions, employment assistance and other benefits, making slightly less than one-third of the entire expense. The second highest expense of the budget were health and personal social services. Hospital and medical services covered AED 53 billion, or about 15% of the budget. On the other hand, education cost UAE AED 38 billion, comprising nearly 12% of the entire budget. The government spent approximately 7% of revenue on debt, and just about similar amounts were spent on defence, which was AED 22 billion, and law and order, which comprised AED 17 billion.

Expenditure on housing, transport and industry came to a total of AED 37 billion. Lastly, other spending reported for AED 23 billion.
            """
        )

    # 初始化生成器
    generator = GraphGenerator()

    # 执行API调用
    result = generator.call_gpt_and_generate(
        initial_instruction=initial_instruction,
        requirement=requirement,
        # sample_answer=sample_answer,
        student_answer=student_answer,
        image_path="data/pie2.png",  # 或指定图片路径
        output_format="json"
    )

    print(extract_table_from_image_deplot("data/pie2.png"))

    # 处理结果
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Generated Data:")
        print(json.dumps(result, indent=2))
