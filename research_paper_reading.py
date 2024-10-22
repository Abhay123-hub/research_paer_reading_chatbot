import streamlit as st
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_openai import ChatOpenAI
import os
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
import time

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "enter your openai api"
# Initialize session state for API wrappers and prompts
if "vector" not in st.session_state:
    st.session_state.api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    st.session_state.wiki = WikipediaQueryRun(api_wrapper=st.session_state.api_wrapper_wiki)  # Wikipedia tool
    st.session_state.api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    st.session_state.arxiv = ArxivQueryRun(api_wrapper=st.session_state.api_wrapper_arxiv)  # Arxiv web page tool
    
    # Pull the correct OpenAI tools prompt from hub
    st.session_state.prompt_init = hub.pull("hwchase17/openai-functions-agent")
    
    # Initialize the prompt using a valid template
    st.session_state.prompt = ChatPromptTemplate.from_messages(st.session_state.prompt_init.messages)

# Streamlit UI setup
st.title("Research Paper Reading Chatbot 📜🔍🤖💡")

# Set up LLM and tools
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [st.session_state.arxiv, st.session_state.wiki]

# Create the agent and agent executor using the correct prompt
agent = create_openai_tools_agent(llm, tools, st.session_state.prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Get user input for the prompt
prompt = st.text_input("Write your prompt here")

# If prompt is provided, process the input
if prompt:
    with st.spinner("Processing..."):
        start_time = time.process_time()
        response = agent_executor.invoke({"input": prompt})
        elapsed_time = time.process_time() - start_time
        st.write(f"Response time: {elapsed_time:.2f} seconds")

        # Display the output or show an error if none found
        if 'output' in response:
            st.write(response['output'])
        else:
            st.write("No output found in the response")


