from graphviz import Digraph

# Define the categories and their respective tools
information_retrieval_tools = [
    "TavilySearchResults",
    "google_search",
    "Search_The_Internet (Google, Brave Search, DuckDuckGo, Tavily, Searxng)",
    "Wikipedia_Search",
    "Search_Youtube",
    "Get_Youtube_Captions",
    "Scrape_Web_URL",
]

utility_tools = [
    "python_repl",
    "Datetime",
]

knowledge_tools = [
    "Chroma_DB_Search",
]


# Function to generate HTML table rows for the tools node label
def generate_table_rows(category, tools):
    tools_list = "<br/>".join(tools)
    return f'<TR><TD bgcolor="#e0e0e0"><B>{category}</B></TD><TD>{tools_list}</TD></TR>'


# Generate the tools node label
table_rows = [
    generate_table_rows("Information Retrieval Tools", information_retrieval_tools),
    generate_table_rows("Utility Tools", utility_tools),
    generate_table_rows("Knowledge Base Tools", knowledge_tools),
]
tools_label = (
    f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
    f'<TR><TD COLSPAN="2"><B>Tools</B></TD></TR>'
    f"{''.join(table_rows)}"
    f"</TABLE>>"
)

# Create the main graph with enhanced attributes
dot = Digraph(
    comment="Skills AI Backend System",
    format="png",
    graph_attr={
        "rankdir": "TB",  # Top-to-bottom layout
        "splines": "spline",  # Smooth edges
        "bgcolor": "#f0f0f0",  # Light gray background
        "label": "Skills AI Backend System\nBuilt with Langchain, LangGraph, and FastAPI",
        "labelloc": "t",  # Label at top
        "fontsize": "16",
        "fontcolor": "#333333",
    },
    node_attr={
        "shape": "box",
        "style": "filled,rounded",  # Rounded corners
        "fontname": "Helvetica",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "10",
    },
)

# Subgraph for API Layer
with dot.subgraph(name="cluster_api") as api:
    api.attr(label="API Layer (FastAPI)", style="filled", color="#d9e6f2")
    api.node(
        "api_server",
        label="API Server\n(FastAPI, OAuth2, JWT)",
        fillcolor="#a3c2e0",
        shape="rect",
    )
    api.node(
        "redis",
        label="Redis\n(Caching)",
        fillcolor="#ff9999",
        shape="cylinder",
    )
    api.node(
        "db",
        label="Database\n(PostgreSQL)",
        fillcolor="#99ff99",
        shape="cylinder",
    )

# Subgraph for Langchain Components
with dot.subgraph(name="cluster_langchain") as langchain:
    langchain.attr(label="Langchain Components", style="filled", color="#e6f2d9")
    langchain.node(
        "SystemPrompt",
        label="System Prompt",
        shape="parallelogram",
        fillcolor="#ffff99",
        color="#cccc00",
    )
    langchain.node(
        "chatbot",
        label="Chatbot\nLLM: Gemini-2.0-pro-exp-03-25\n(with Chat Analysis)",
        fillcolor="#4d79ff",
        color="#1a53ff",
        fontcolor="white",
        shape="rect",
    )
    langchain.node(
        "tools",
        label=tools_label,
        fillcolor="#b3d9ff",
        color="#4d79ff",
        width="4",
        height="2",
    )

with dot.subgraph(name="cluster_external") as external:
    external.attr(label="External Services", style="filled", color="#f2d9e6")
    external.node(
        "s3_bucket",
        label="S3 Bucket",
        shape="folder",
        fillcolor="#ffcc99",
        style="filled,rounded",
    )

# Nodes outside subgraphs (entry/exit points)
dot.node(
    "START",
    label="User Input\n(via API)",
    shape="circle",
    color="#ff4d4d",
    fillcolor="#ff9999",
    fontcolor="white",
)
dot.node(
    "END",
    label="Final Output\n(to API)",
    shape="circle",
    color="#ff4d4d",
    fillcolor="#ff9999",
    fontcolor="white",
)

# Define edges
dot.edge("START", "api_server", label="HTTP Request", color="#666666")
dot.edge("api_server", "chatbot", label="Process Input", color="#666666")
dot.edge("api_server", "redis", label="Cache Access", color="#666666", style="dashed")
dot.edge("api_server", "db", label="DB Operations", color="#666666", style="dashed")
dot.edge("SystemPrompt", "chatbot", label="Initializes", color="#666666")
dot.edge("chatbot", "tools", label="If Tools Needed", color="#00cc00", style="dashed")
dot.edge("tools", "chatbot", label="Tool Results", color="#00cc00", style="dashed")
dot.edge("chatbot", "api_server", label="Response", color="#666666")
dot.edge("api_server", "END", label="HTTP Response", color="#666666")
dot.edge(
    "api_server",
    "s3_bucket",
    label="Image Storage/Retrieval",
    color="#666666",
    style="dashed",
)

# Render the graph
dot.render("images/enhanced_skills_ai_graph", view=True)
print(
    "Enhanced Graphviz graph has been generated and saved as 'images/enhanced_skills_ai_graph.png'."
)
