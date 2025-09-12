from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

search_tool = DuckDuckGoSearchRun(region="us-en") 

@tool
def search_tool_wrapper(query: str) -> dict:
    """
    Search the web for a query and return top 5 concise results.
    """
    result = search_tool.run(query)
    # Split results into lines and pick first 5
    lines = result.split("\n")
    top_results = lines[:5]
    return {"query": query, "top_results": top_results}



@tool

def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation=='add':
            result= first_num+second_num

        elif operation=='sub':
            result= first_num-second_num

        elif operation=='mul':
            result= first_num*second_num

        elif operation=='div':
            if second_num==0:
                return {'error': 'division by 0 is not allowed'}

            return first_num/second_num
        
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        

        return {'first num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}
    
    except Exception as e:
        return {'error': str(e)}
    


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=QEXG4A9MN9Z2LZCC"
    r = requests.get(url)
    try:
        data = r.json()
        price = data["Global Quote"]["05. price"]
        return {"symbol": symbol, "price": price}
    except Exception as e:
        return {"error": str(e)}


tools = [search_tool_wrapper, calculator, get_stock_price]


llm_with_tools = llm.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    system_prompt = SystemMessage(
        content=(
            "You are a helpful AI assistant. You have access to the following tools:\n"
            "1) search_tool → use this to answer questions about current news or general info.\n"
            "2) calculator → use this to solve math problems.\n"
            "3) get_stock_price → use this to fetch the latest stock prices.\n\n"
            "Whenever a user asks a question that requires any of these tools, call the tool instead of answering from memory."
        )
    )

    # Prepend the system message
    messages = [system_prompt] + state.get("messages", [])

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



tool_node=ToolNode(tools)

conn=sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)