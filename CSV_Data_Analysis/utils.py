##from langchain.agents import create_pandas_dataframe_agent #This import has been recently replaced by the below
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
#from langchain.llms import OpenAI
#New import from langchain, which replaces the above
from langchain_openai import OpenAI

def query_agent(data, query):

    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    llm = OpenAI()
    
    # Create a Pandas DataFrame agent and allow dangerous code execution.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

    # Execute the query on the DataFrame and return the result.
    result = agent.invoke(query)
    return result
