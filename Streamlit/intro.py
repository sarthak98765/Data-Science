import streamlit as st

# st.text("Hello everyone")

st.set_page_config("Titanic")   #should be first command in streamlit script
st.write("hi ml **enthusiast**")

st.title("This is title")
st.header("This is header")
st.subheader("This is subheader")

x = '''def func():
    print(np.arange(10))'''
st.code(x , language = "python")


# Markdown
st.markdown("# This is a markdown")