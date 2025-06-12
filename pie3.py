import os
from io import BytesIO
import streamlit as st
import openai
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# 加载 DePlot 模型（只加载一次）
deplot_processor = Pix2StructProcessor.from_pretrained('google/deplot')
deplot_model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-3d63de1f971640958d6cbbe98ae670b7"  # 替换为你的实际API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # 确认最新API地址


def custom_clean_deplot_text(raw_text: str) -> str:
    messages = [
        {"role": "system", "content": deplot_processor(raw_text)},
        {"role": "user", "content": f"Task Context: {initial_instruction}"},
        {"role": "user", "content": f"Official Requirement: {requirement}"},
        {"role": "user", "content": f"""
                        [BEGIN STUDENT ANSWER]
                        {student_answer}
                        [END STUDENT ANSWER]
                        """}
    ]


def parse_txt_to_dict(txt_content: str) -> Dict[str, Optional[float]]:
    lines = txt_content.strip().splitlines()
    data = {}
    for line in lines[1:]:  # 跳过标题
        if "|" in line:
            category, value = [part.strip() for part in line.split("|", 1)]
            if value.lower() == "null":
                data[category] = None
            else:
                percent_match = re.search(r"([\d.]+)%", value)
                if percent_match:
                    data[category] = float(percent_match.group(1))
    return data


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

    def call_gpt_and_generate(self, initial_instruction: str, requirement: str, student_answer: str,
                              image_path: Optional[str] = None, model: str = "deepseek-chat",
                              output_format: str = "json") -> Dict:
        try:
            system_content = """
            You are a data extractor analyzing student-written IELTS Academic Task 1 answers that describe statistical charts.

            1. Identify all explicitly mentioned categories related to data (e.g., government spending, population segments, etc.).

            2. For each category:
               - If a percentage value is mentioned (e.g., "12%"), extract it directly.
               - If only an absolute value is given (e.g., "38 billion"), and the total value is not mentioned, you may estimate the percentage using a logical assumption (e.g., based on 100%), but avoid inventing specific totals. The result must still be expressed as a percentage.
               - If **no numerical value** is mentioned at all, **ignore this category completely**. Do not output it, even as `null`.

            3. If multiple categories are combined into a single line (e.g., "housing, transport and industry"), **keep them as a combined category**:
               - Do not split them into individual categories.
               - Example: If "housing, transport and industry" is mentioned with 37 billion, it should appear as:
                 - housing, transport and industry | estimated %

            4. For ambiguous or generic categories like "other spending", "entire households", or "various", **include them as their own category**:
               - Example: "other spending" should be listed as:
                 - other spending | estimated %

            5. If a category is described **relative to another category** (e.g., "about the same as X", "double of Y", "slightly less than Z"):
               - Estimate the percentage based on the described category.
               - Example:
                 - "about the same as health" → if health is 15%, estimate it as 15%.
                 - "double of defence" → if defence is 7%, estimate it as 14%.
                 - "slightly less than education" → if education is 12%, estimate it as around 10-11%.

            6. Handle **vague or comparative descriptions** smartly:
               - Phrases like "almost the same as", "slightly higher than", "a bit lower than" should be translated to estimated percentages:
                 - "almost the same as" → ±1~2%
                 - "slightly higher than" → +2~5%
                 - "slightly lower than" → -2~5%
               - Example:
                 - "almost the same as single people" → if single people is 24%, estimate it as 23% or 25%.
                 - "slightly higher than aged couples" → if aged couples is 9%, estimate it as 11%.

            7. Output all values as **percentages**:
               - Final output must only contain percentages (e.g., `15%`).
               - Do not include raw units like “billion” or “AED”.
               - If the category is mentioned but has no numerical value, **completely remove it from the output**.

            8. Your output must be in plain text, using the following structure:

                TITLE | <concise inferred chart title>

                <Category 1> | <percentage>
                <Category 2> | <percentage>
                ...

                * Do not include categories without values.
                * Do not include commentary, explanation, code, or extra formatting.
                * Do not invent or guess any categories that are not clearly stated in the text.


            """

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Task Context: {initial_instruction}"},
                {"role": "user", "content": f"Official Requirement: {requirement}"},
                {"role": "user", "content": f"""
                    [BEGIN STUDENT ANSWER]
                    {student_answer}
                    [END STUDENT ANSWER]
                    """}
            ]

            # 提取并添加颜色调色板
            palette = self.extract_color_palette(image_path)

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
            )
            print(response.choices[0].message.content)

            return self._process_response(response, output_format, image_path, palette)

        except Exception as e:
            print(f"API Error: {str(e)}")
            return {"error": str(e)}

    def extract_table_from_image_deplot(self, image_path: str) -> str:
        # 原始 DePlot 调用
        image = Image.open(image_path).convert("RGB")
        inputs = deplot_processor(images=image, text="Generate underlying data table of the figure below:",
                                  return_tensors="pt")
        predictions = deplot_model.generate(**inputs, max_new_tokens=512)
        raw_text = deplot_processor.decode(predictions[0], skip_special_tokens=True)

        system_content = """
            You are a data extractor for chart analysis. Your task is to:

            1. Clean and structure the raw DePlot data to be clear and readable.
            2. The format should be:

                TITLE | <concise inferred chart title>

                <Category 1> | <percentage or null>
                <Category 2> | <percentage or null>
                ...

            3. Ensure:
                - Categories are aligned with the original DePlot order.
                - Percentages are preserved, or marked as `null` if missing.
                - No additional commentary or explanation is included.
                - The final output is in plain text format.
            """

        initial_instruction = "Process the raw DePlot output to match the required format."
        requirement = "Transform the raw data into the specified format, with categories and percentages cleanly aligned."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Task Context: {initial_instruction}"},
            {"role": "user", "content": f"Official Requirement: {requirement}"},
            {"role": "user", "content": f"""
            [BEGIN DEPLOT DATA]
            {raw_text}
            [END DEPLOT DATA]
            """}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.0,
            max_tokens=1500
        )
        print(f"\n{response.choices[0].message.content}")

        cleaned_text = response.choices[0].message.content.strip()
        return cleaned_text

    def find_best_match_batch(self, target: str, candidates: list, cutoff=0.85) -> Optional[str]:
        """
        使用AI模型批量对比目标类别和所有学生类别，返回最接近的匹配项
        """
        target = target.lower()  # 转为小写字母，标准化输入
        candidates = [candidate.lower() for candidate in candidates]  # 确保所有候选项都是小写

        # 修改后的 GPT 批量比较的 prompt
        system_content = """
                You are a semantic comparison model. Your task is to compare a target phrase with a list of candidate phrases and return the most semantically similar phrase.
                You will provide a score for each phrase in the format: 'phrase_name: score' (e.g., 'single parents: 0.9').
                Return only the phrase with the highest score without changing the original phrasing of the candidates.
                """
        candidate_description = "\n".join([f"Candidate {i + 1}: {candidate}" for i, candidate in enumerate(candidates)])

        initial_instruction = f"""
                Compare the following target phrase with a list of candidate phrases. The target phrase is: "{target}"
                The candidates are:
                {candidate_description}
                For each candidate, provide a similarity score between 0 and 1, where 1 means they are very similar and 0 means they are completely different.
                Only provide the phrase with the highest similarity score, in the format 'phrase_name: score'.
                Do not modify or alter the original phrasing of the candidate phrases in any way.
                """

        requirement = """
                Please return only the matching phrase with the highest similarity score, without any extra information or modifications.
                """

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Task Context: {initial_instruction}"},
            {"role": "user", "content": f"Official Requirement: {requirement}"},
            {"role": "user", "content": f"""
                    [BEGIN COMPARISON]
                    Target phrase: {target}
                    {candidate_description}
                    [END COMPARISON]
            """}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=100,
            temperature=0
        )
        print(f"\n{response.choices[0].message.content}")  # 打印响应内容，以便查看格式

        # 解析 AI 的响应并找到相似度最高的匹配项
        similarity_scores = response.choices[0].message.content.strip().split('\n')
        best_match = None
        highest_similarity = 0

        # 遍历分数，选择最佳匹配
        for score in similarity_scores:
            if score:
                # 检查返回格式，确保存在 ':' 分隔符
                if ':' in score:
                    phrase, score_value = score.split(':')
                    try:
                        score_value = float(score_value.strip())
                        # 仅保留相似度高于阈值的类别
                        if score_value > highest_similarity and score_value >= cutoff:
                            highest_similarity = score_value
                            best_match = phrase.strip()  # 确保提取的是类别名，而非对比信息
                    except ValueError:
                        continue  # 如果无法转换为浮动数值，跳过此项
                else:
                    print(f"Skipping invalid format: {score}")  # 打印没有符合预期格式的响应

        return best_match

    def compare_and_generate_json(self, deplot_txt: str, student_txt: str, palette, title="Generated Chart") -> Dict:
        deplot_data = parse_txt_to_dict(deplot_txt)
        student_data = parse_txt_to_dict(student_txt)

        categories = []
        values = []
        categories_match = []

        # 遍历 DePlot 的顺序
        for cat, deplot_val in deplot_data.items():
            best_match = self.find_best_match_batch(cat, list(student_data.keys()))  # 使用批量对比
            if best_match:
                categories_match.append(best_match)
                categories.append(cat)
                values.append(student_data[best_match])
        for cat in student_data:
            if cat not in categories and cat not in categories_match:
                categories.append(cat)
                values.append(student_data[cat])
        total = sum(values)
        if total < 100:
            categories.append("Missing")
            values.append(round(100 - total, 1))
            missing_index = len(categories) - 1
        else:
            missing_index = None

        json_output = {
            "title": title,
            "categories": categories,
            "x_label": "",
            "y_label": "Percentage",
            "series": [{"values": values}],
            "chart_type": "pie",
            "style": {
                "color_palette": palette,
            }
        }
        if missing_index is not None:
            json_output["style"]["missing_index"] = missing_index
        return json_output

    def _process_response(self, response, output_format: str, image_path, palette) -> Dict:
        """
        响应处理方法
        """
        try:
            student_txt = response.choices[0].message.content
            deplot_txt = self.extract_table_from_image_deplot(image_path)
            data = self.compare_and_generate_json(deplot_txt, student_txt, palette)
            # 调试：将完整返回内容写入 debug.txt 便于查看
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("=== Raw Response Content 已保存到 debug.txt ===")

            if output_format == "json":
                self._plot_from_json(data)
                return data
            return self._safe_execute_code(str(data))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _increment_counter(self):
        """递增计数器并保存到文件中"""
        self.data_counter += 1
        with open(self.counter_file_path, "w") as f:
            f.write(str(self.data_counter))

    def _plot_from_json(self, data: Dict):
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
        missing_indices = data.get("style", {}).get("missing_index")

        # 如果是整数，转成列表（兼容旧代码）
        if isinstance(missing_indices, int):
            missing_indices = [missing_indices]
        elif not isinstance(missing_indices, list):
            missing_indices = []

        # 遍历每个索引，进行高亮处理
        for idx in missing_indices:
            if 0 <= idx < len(wedges):
                wedges[idx].set_facecolor((1.0, 0.3, 0.3, 0.4))  # 半透明红色
                wedges[idx].set_hatch('//')  # 斜线填充
                wedges[idx].set_edgecolor('black')

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

    #     student_answer = (
    #         """
    #         The pie chart inspects the different family types living in poor conditions in the UK in 2002.
    # At a glance, in the given year, 14% of the entire households in the country were in circumstances of poverty. In comparison to the couples, singles struggled more. Talking about people with children, single parents presented the maximum percentage of 26% amongst all the specified categories, whereas couples with children reported a comparatively lesser percentage of 15%.
    # As far as the people with no children are concerned, single people were of the hefty percentage, 24%, almost the same number for single people with children. On the contrary, merely 9% of the couples without any children agonized from poverty in 2002. Coming to aged people, singles had a somewhat higher percentage in comparison to couples.
    #         """
    #     )
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

    # 处理结果
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Generated Data:")
        print(json.dumps(result, indent=2))
