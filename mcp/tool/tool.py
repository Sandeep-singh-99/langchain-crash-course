from langchain.tools import tool

@tool
def calculator(input: str) -> str:
    """
    Add two numbers
    Input format: "a,b"
    Example: "10,20"
    """

    a, b = input.split(",")
    return str(int(a) + int(b))

@tool
def get_notes(input: str) -> str:
    """
    Return saved notes content
    """
    return "MCP allows AI models to safely use external tools."