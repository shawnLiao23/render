import os
import streamlit as st
from PIL import Image
from pie import GraphGenerator

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

    # 页面布局
    st.markdown("""
        <style>
        .block-container {
            max-width: 1500px;  /* 设置最大宽度 */
            padding-left: 2rem;  /* 设置左侧内边距 */
            padding-right: 2rem;  /* 设置右侧内边距 */  
        }
    """, unsafe_allow_html=True)

    # 创建左右两栏
    left_col, right_col = st.columns([1,1])

    with right_col:
        st.header("📈 Student Graph")
        placeholder = st.empty()
        placeholder.markdown("Your graph would be displayed after the generation.")

    with left_col:
        st.header("📋 Requirement") # 展示题目要求
        requirement = (
            """The pie chart gives information on UAE government spending in 2000. The total budget was AED 315 billion.
            Summarize the information by selecting and reporting the main features, and make comparisons where relevant.
            Write at least 150 words."""
            )
        st.markdown(requirement)

        # 显示原图
        st.subheader("📊 Original Image")
        img = Image.open("data/pie2.png")
        st.image(img, caption="Original Graph", use_container_width=True)

        # 学生作文输入框
        st.subheader("✍️ Student Answer")
        student_answer = st.text_area("Please write your answer here:", height=300)
        text_path = os.path.join(generator.data_save_folder, f"answer{generator.data_counter}.txt")
        # 保存学生答案为.txt文件
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(student_answer)

        # 按钮触发生成图像
        if st.button("🚀 Generate Graph from Student Answer"):
            if not student_answer.strip():
                st.warning("Please enter the student's answer before generating.")
            else:
                with st.spinner("Generating student-based chart..."): # 调用主函数
                    placeholder.empty()
                    with right_col:
                         result = generator.call_gpt_and_generate(
                             initial_instruction=initial_instruction,
                             requirement=requirement,
                             student_answer=student_answer,
                             image_path="data/pie2.png",
                             output_format="json" )
            if "error" in result:
                 st.error(f"Generation failed: {result['error']}")
            else:
                 st.success("✅ Student-based chart generated successfully!")

    with right_col:
        st.header("💡 Writing Suggestions")
        st.markdown("to be developed...")
