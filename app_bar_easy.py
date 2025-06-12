import os
from pprint import pprint

import streamlit as st
from PIL import Image
from bar import GraphGenerator

def show():
    # 初始化
    generator = GraphGenerator()

    # 输入instruction
    initial_instruction = (
        "Now I'll send you the Requirement, graph and Sample answer of the first Writing question of IELTS Academic. "
        "You need to learn how to reverse generate the graph according to the requirement and given answer which "
        "describes the graph by the materials I give to you."
    )

    # 页面标题
    st.title("🎓 IELTS Task 1 Graph Generation System")

    # 页面布局样式设置
    st.markdown("""
        <style>
        .block-container {
            max-width: 1500px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 创建左右两栏
    left_col, right_col = st.columns([1, 1])

    with right_col:
        st.header("📈 Student Graph")
        placeholder = st.empty()
        placeholder.markdown("Your graph would be displayed after the generation.")

    with left_col:
        st.header("📋 Requirement")
        requirement = (
            "The bar chart below shows the total number of minutes (in billions) of telephone calls in Australia,"
            "divided into three categories, from 2001- 2008. Summarise the information by selecting and reporting the"
            "main features and make comparisons where relevant. Write at least 150 words."
        )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/bar.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        st.subheader("👉 Recommend Structure")
        structure = ("The given chart depicts the time Australian residents spent on varying types of telephone calls between 20__ and 2008."
    "Local fixed line calls were the highest throughout this period, upsurging from 72 billion minutes to under 90 billion in 20__. "
    "Following year, this figure peaked at 9_ billion. Post this, by 2008, it had a downtrend and fell back to the figure of 2001. "
    "Both national and international fixed line calls grew gradually from ___ billion to 61 billion toward the end of the period in question."
    "However, the progress decelerated over the last two years. Also, dramatic growth can be seen in mobile calls from 2 billion to _6 billion"
    "minutes. This increase was specifically noticed between ____ and 2008. During this time, the mobile phone’s use got tripled. In 2008,"
    "although local fixed line calls were still popular, the gap between these three categories narrowed significantly over the second half of this period."
        )
        st.markdown(structure)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please follow the structure and write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")

        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        tsv_text = """Year \t Local fixed line calls \t National and international fixed line calls \t Mobile calls \n 2001 \t 72 \t 38 \t 3 \n 2002 \t 78 \t 40 \t 6 \n 2003 \t 83 \t 42 \t 10 \n 2004 \t 88 \t 45 \t 12 \n 2005 \t 90 \t 47 \t 15 \n 2006 \t 85 \t 50 \t 23"""
        chartvlm_data = generator.parse_chartvlm_csv(tsv_text)

        # 按钮触发生成图像
        if st.button("🚀 Generate Graph from Student Answer"):
            if not student_answer.strip():
                st.warning("Please enter the student's answer before generating.")
            else:
                with st.spinner("Generating student-based chart..."):
                    placeholder.empty()
                    with right_col:
                        result = generator.call_gpt_and_generate(
                            initial_instruction=initial_instruction,
                            requirement=requirement,
                            student_answer=student_answer,
                            image_path="data/bar.png",
                            output_format="json",
                            chartvlm_data=chartvlm_data
                        )

                        if result is None:
                            st.error("❌ Generation failed: ChartVLM or GPT did not return usable data.")
                        else:
                            st.success("✅ Generation complete. Chart shown above.")


    with right_col:
        st.header("💡 Writing Suggestions")
        st.markdown("to be developed...")
