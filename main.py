import streamlit as st

# 页面配置只能设置一次，在主入口文件中
st.set_page_config(page_title="📚 IELTS Writing Tool", layout="wide")

import app_bar
# import app_bar_easy
# import app_bar2
# import app_bar2_easy
# import app_bar3
# import app_bar3_easy
# import app_pie
# import app_pie_easy
# import app_pie2
# import app_pie2_easy

# 页面选择
page = st.sidebar.selectbox("Please choose a question", ["Bar", "Bar easy","Bar2", "Bar2 easy", "Bar3", "Bar3 easy", "Pie", "Pie easy","Pie2","Pie2 easy"])

if page == "Bar":
    app_bar.show()
# if page == "Bar easy":
#     app_bar_easy.show()
# if page == "Bar2":
#     app_bar2.show()
# if page == "Bar2 easy":
#     app_bar2_easy.show()
# if page == "Bar3":
#     app_bar3.show()
# if page == "Bar3 easy":
#     app_bar3_easy.show()
# if page == "Pie easy":
#     app_pie_easy.show()
# if page == "Pie2":
#     app_pie2.show()
# if page == "Pie2 easy":
#     app_pie2_easy.show()
# elif page == "Pie":
#     app_pie.show()
