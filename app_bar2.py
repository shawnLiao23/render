import os
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
            "The chart below shows the amount of leisure time enjoyed by men and women of different employment status."
"Write a report for a university lecturer describing the information shown below."
"Leisure time in a typical week in hour: by sex and employment status, 1998-99."
        )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/bar2.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")

        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        tsv_text = """Characteristic \t Male \t Female \n Employed (Full Time) \t 44 \t 38 \n Employed (Part Time) \t 85 \t 40 \n Unemployed \t 78 \t 78 \n Retired \t 83 \t 78 \n Housewives \t 50 \t 50 \n"""
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
                            image_path="data/bar2.png",
                            output_format="json",
                            chartvlm_data=chartvlm_data
                        )

                if "error" in result:
                    st.error(f"Generation failed: {result['error']}")
                else:
                    st.success("✅ Student-based chart generated successfully!")

    with right_col:
        st.header("💡 Writing Suggestions")
        st.markdown("to be developed...")
