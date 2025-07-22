import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

#==== Streamlit Configuration ===
groq_api_key = st.secrets["GROQ_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]

# ==== Model Setup ====
llm = ChatGroq(api_key=st.secrets.get("GROQ_API_KEY"), model_name="llama3-70b-8192")
search = TavilySearchResults(max_results=2)
parser = StrOutputParser()

# ==== Page Layout ====
st.title("High-Performance Computing (HPC) Sales Assistant Agent")
st.markdown("🚀 Sales Engineer Assistant powered by **Groq**, **LangChain**, and **Tavily**")

# ==== Input Form ====
with st.form("company_info_form", clear_on_submit=True):
    product_name = st.text_input("**Product Name**")
    company_url = st.text_input("**Company URL** (e.g., https://www.supermicro.com/en/)")
    product_category = st.text_input("**Product Category** (e.g., Servers, Storage)")
    competitors_url = st.text_input("**Competitor's URL**")
    value_proposition = st.text_input("**Value Proposition**")
    target_customer = st.text_input("**Target Customer** (e.g., Senior IT Director)")

    submit = st.form_submit_button("Generate Insights")

# ==== Logic ====
if submit:
    if not product_name or not company_url:
        st.warning("⚠️ Please enter both a Product Name and Company URL.")
    else:
        with st.spinner("🔍 Generating HPC insights..."):
            try:
                company_info = search.invoke(company_url)

                prompt = """
                You are an expert in High-Performance Computing (HPC) solutions, specializing in Supermicro products.
                Assist sales engineers by generating persuasive product insights and comparisons.

                Responsibilities:
                1. Product Expertise – Understand Supermicro servers, storage, networking.
                2. Competitive Analysis – Compare against HPE on performance, cost, scalability.
                3. Customer Support – Offer custom-fit product recommendations.

                🔍 Company Info: {company_information}
                💼 Product Name: {product_name}
                🏷️ Product Category: {product_category}
                🔄 Competitor's URL: {competitors_url}
                🎯 Target Customer: {target_customer}
                💡 Value Proposition: {value_proposition}

                Provide actionable insights a sales engineer can use in a pitch.
                """

                # Create chain
                prompt_template = ChatPromptTemplate.from_template(prompt)
                chain = prompt_template | llm | parser

                # Fill template
                result = chain.invoke({
                    "company_information": company_info,
                    "product_name": product_name,
                    "product_category": product_category,
                    "competitors_url": competitors_url,
                    "value_proposition": value_proposition,
                    "target_customer": target_customer
                })

                st.markdown("### 🧠 AI-Generated HPC Sales Insights")
                st.markdown(result)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
