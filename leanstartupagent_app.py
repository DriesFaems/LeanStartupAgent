import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.agents import AgentType

st.title("LeanStartupAgent")

#define company and project

open_api_key = st.text_input('Enter your open api key. This information is not recorded or stored in any way', type = "password")
serpapi_api_key = st.text_input('Enter your serpapi api key. This information is not recorded or stored in any way', type = "password")

job_to_be_done = st.text_input('Describe the core job to be done')
customer_description = st.text_input('Describe your customer segment')

clicked = st.button('Click me')

if clicked:
    st.write('Button clicked! Performing an operation...')
    # Place the code that should only execute after click here
    llm = OpenAI(openai_api_key=open_api_key, temperature=0.5)
    hardest_template = PromptTemplate(
      input_variables = ['customer_description','job_to_be_done'], 
      template='You are {customer_description}. What is the hardest part about {job_to_be_done}?')
    hardest_chain = LLMChain(llm=llm, prompt=hardest_template, verbose=True, output_key='hardest_part')
    hardest = hardest_chain.run(customer_description=customer_description, job_to_be_done=job_to_be_done)
    st.markdown('**Agent core painpoint =**')
    st.write(hardest)
    value_proposition_template = PromptTemplate(
      input_variables = ['hardest_part'], 
      template='Create a unique value proposition for a startup that solves the problem of {hardest_part}')
    value_proposition_chain = LLMChain(llm=llm, prompt=value_proposition_template, verbose=True, output_key='value_proposition')
    valueproposition = value_proposition_chain.run(hardest_part=hardest)
    st.markdown('**Agent Value proposition =**')
    st.write(valueproposition)
    canvas_template = PromptTemplate(
      input_variables = ['value_proposition'], 
      template='Provide the value proposition canvas for {value_proposition}. You should describe the following elements: customer jobs, custome pains, customer gains, products and services, pain relievers, gain creators')
    canvas_chain = LLMChain(llm=llm, prompt=canvas_template, verbose=True, output_key='canvas')
    canvas = canvas_chain.run(value_proposition=valueproposition)
    st.markdown('**Agent description of value proposition canvas =**')
    st.write(canvas)
    search = SerpAPIWrapper(serpapi_api_key = serpapi_api_key)
    search_tool = [Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search")]
    # Prompt competitor identification based on SerpAPPI
    search = SerpAPIWrapper(serpapi_api_key = serpapi_api_key)
    self_ask_with_search = initialize_agent(search_tool, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    searchstring = "Search three existing startup companies that have a description that is very similar to the following description:" + valueproposition + "Provide for each company the city location. Ensure that the output matches the expected format, including the presence of the finish_string."
    competitors = self_ask_with_search.run(searchstring)
    st.markdown('**Agent identification of competitors =**')
    st.write(competitors)
    competitors_list = competitors.split('), ')
    # prompting hypotheses
    hypotheses_template = PromptTemplate(
      input_variables = ['value_proposition', 'canvas'], 
      template='Generate a list of hypotheses for testing the following value proposition: {value_proposition}. You should rely on information from the value proposition canvas: {canvas}. Include at least 3 hypotheses. Each hypothesis should focus on the feasibility or viability of the business model. Here are some potential topics for hypotheses testing: willingness to pay, preferred distribution model, ultimate customer segmet, etc.')
    hypotheses_chain = LLMChain(llm=llm, prompt=hypotheses_template, verbose=True, output_key='hypotheses')
    hypotheses = hypotheses_chain.run(value_proposition=valueproposition, canvas=canvas)
    st.markdown('**Agent hypotheses =**')
    st.write(hypotheses)
else:
    st.write('Please click the button to perform an operation')