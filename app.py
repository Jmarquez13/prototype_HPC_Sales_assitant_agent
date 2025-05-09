import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults


# Model and Agent tools
llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="compound-beta")
search = TavilySearchResults(max_results=2)
parser = StrOutputParser()
# tools = [search] # add tools to the list

# Page Header
st.title("HPC Assistant Agent")
st.markdown("HPC Assistant Agent Powered by Groq.")


# Data collection/inputs
with st.form("company_info", clear_on_submit=True):

    product_name = st.text_input("**Product Name** (What product are you selling?):")
    
    company_url = st.text_input(
        "**Company URL** (The URL of the company you are targeting? ex.https://www.supermicro.com/en/ ):"
    )
    
    product_category = st.text_input(
        "**Product Category** (e.g., 'Servers' & 'Storage'):"
    )
    
    competitors_url = st.text_input("**Competitors URL** (ex. https://www.hpe.com):")
    
    value_proposition = st.text_input(
        "**Value Proposition** (A sentence summarizing the product’s value):"
    )
    
    target_customer = st.text_input(
        "**Target Customer** (Name of the person you are trying to sell to.) :"
    )

    # For the llm insights result
    company_insights = ""

    # Data process
    if st.form_submit_button("Generate Insights"):
        if product_name and company_url:
            st.spinner("Processing...")

            # Use search tool to get Company Information
            company_information = search.invoke(company_url)
            print(company_information)

           # TODO: Create prompt <=================
            prompt = f"""
            You are an expert in High-Performance Computing (HPC) solutions, specializing in Supermicro products. Your primary goal is to assist sales engineers in providing comprehensive and persuasive product information, comparisons, and insights to customers.
            Key Responsibilities:
            1. Product Knowledge:
                - Become a subject matter expert on Supermicro's entire product range, including servers, storage, networking solutions, and components.
                - Stay updated on the latest product releases, features, and specifications.
                - Understand the technical nuances of Supermicro products and their competitive advantages.
            2. Competitive Analysis:
                - Conduct in-depth analysis of HPE's HPC offerings.
                - Identify key differentiators between Supermicro and HPE products, focusing on performance, efficiency, scalability, and cost-effectiveness.
                - Develop compelling arguments to position Supermicro as the superior choice.
            3. Customer Insights:
                - Anticipate customer needs and questions.
                - Provide tailored product recommendations based on specific customer requirements.
                - Offer valuable insights into network solutions and their impact on overall HPC performance.
            4. Sales Support:
                - Assist sales engineers in crafting persuasive sales pitches and presentations.
                - Provide quick and accurate responses to customer inquiries.
                - Generate creative ideas for marketing materials and sales collateral.
            Company info: {company_information}
            Product name: {product_name}
            Competitors URL: {competitors_url}
            Product category: {product_category}
            Value proposition: {value_proposition}
            Target customer: {target_customer}
            """
            
            # Company info: {company_information}
            # Product name: {product_name}
            

            # Prompt Template
            prompt_template = ChatPromptTemplate([("system", prompt)])

            # Chain
            chain = prompt_template | llm | parser

            # Result/Insights
            company_insights = chain.invoke(
                {
                    "company_information": company_information,
                    "product_name": product_name,
                    "competitors_url": competitors_url,
                    "product_category": product_category,
                    "value_proposition": value_proposition,
                    "target_customer": target_customer
                }
            )

st.markdown(company_insights)
