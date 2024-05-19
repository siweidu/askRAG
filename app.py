from chain import rag_chain
import streamlit as st



from langserve import add_routes

def main():
    st.set_page_config(page_title='Ask me anything about ...')
    st.header('Ask me anything about Cold Spray of Aluminium')
    user_question = st.text_input('Question about Aluminium cold spray:')
    if user_question:
        response = rag_chain.invoke(user_question)
        st.write(response)




if __name__ == '__main__':
    main()


