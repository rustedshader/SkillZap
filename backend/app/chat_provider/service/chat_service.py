import datetime
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.yahoo_finance_news import (
    YahooFinanceNewsTool,
)
from app.chat_provider.tools.chat_tools import ChatTools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import BraveSearch
from langchain_community.tools import YouTubeSearchTool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


class WebScrapeInput(BaseModel):
    url: str = Field(..., description="URL to scrape data from")


class NoInput(BaseModel):
    pass


# Custom function to query Chroma DB
def retrieve_from_chroma(query: str, vectorstore: Chroma):
    """Retrieve relevant financial documents from Chroma DB with metadata."""
    docs = vectorstore.similarity_search(query, k=3)  # Get top 3 similar documents
    result = ""
    for doc in docs:
        result += f"Snippet: {doc.page_content}\nStart: {doc.metadata['start']}s\nDuration: {doc.metadata['duration']}s\n\n"
    return result.strip()


class ChatService:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        google_search_wrapper: GoogleSearchAPIWrapper,
        tavily_tool: TavilySearchResults,
        google_embedings: GoogleGenerativeAIEmbeddings,
        brave_search: BraveSearch.from_api_key,
    ):
        self.llm = llm
        self.search = google_search_wrapper

        # Initialize existing tools
        self.tavily_tool = tavily_tool
        embeddings = google_embedings
        self.vectorstore = Chroma(
            persist_directory="knowledge_base_db",
            embedding_function=embeddings,
        )
        self.chroma_tool = Tool(
            name="Chroma_DB_Search",
            func=lambda q: retrieve_from_chroma(q, self.vectorstore),
            description="Search the Chroma database for specific financial information.",
        )
        self.yahoo_finance_tool = YahooFinanceNewsTool()

        self.python_repl = PythonREPL()

        self.google_search = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=self.search.run,
        )

        # Initialize ChatTools with multiple search tools
        self.chat_tools = ChatTools(
            duckduckgo_general=DuckDuckGoSearchResults(),
            duckduckgo_news=DuckDuckGoSearchResults(backend="news"),
            searxng=SearxSearchWrapper(searx_host="http://localhost:8080"),
            brave_search=brave_search,
            youtube_search=YouTubeSearchTool(),
        )

        # Define additional tools using ChatTools methods
        self.wikipedia_tool = Tool(
            name="Wikipedia_Search",
            func=self.chat_tools.search_wikipedia,
            description="Search Wikipedia for general information.",
        )

        self.search_web = Tool(
            name="Search_The_Internet",
            func=self.chat_tools.search_web,
            description="Search The Internet to gather information for a given query",
        )

        self.search_youtube = Tool(
            name="Search_Youtube",
            func=self.chat_tools.search_youtube,
            description="Perform comprehesive search on youtube to get the best results and video about given query",
        )

        self.youtube_captions_tool = Tool(
            name="Get_Youtube_Captions",
            func=self.chat_tools.get_youtube_captions,
            description="Retrieve captions and additional info for a list of YouTube video IDs. Input should be a list of video IDs (e.g., ['dQw4w9WgXcQ']).",
        )

        self.scrape_web_url = StructuredTool.from_function(
            name="Scrape_Web_URL",
            func=self.chat_tools.scrape_web_url,
            description="Retrieve data from a specific URL. Input should be a valid URL string.",
            args_schema=WebScrapeInput,
        )

        self.repl_tool = Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=self.python_repl.run,
        )

        self.datetime_tool = StructuredTool.from_function(
            name="Datetime",
            func=lambda: datetime.datetime.now().isoformat(),
            description="Returns the current datetime",
            args_schema=NoInput,
        )

        # Combine all tools into the tools list
        self.tools = [
            self.tavily_tool,
            self.chroma_tool,
            self.yahoo_finance_tool,
            self.wikipedia_tool,
            self.search_web,
            self.search_youtube,
            self.repl_tool,
            self.datetime_tool,
            self.youtube_captions_tool,
            self.google_search,
            self.scrape_web_url,
        ]

        # Initialize state with system message (using the previous system message)
        self.state = {
            "messages": [
                SystemMessage(
                    content="""You are an advanced GenAI skilling assistant designed specifically for Indian youth. Your primary mission is to democratize skill development and empower millions of emerging learners by providing accessible, personalized guidance on vocational training and career development, bridging the gap between current capabilities and rapidly evolving industry demands while ensuring inclusive access across socioeconomic barriers.
                                ---

                                ### Key Responsibilities:
                                - Provide clear, jargon-free explanations of various skills, vocations, and career paths.
                                - Introduce and explain vocational training programs, apprenticeships, and online courses available in India.
                                - Help users assess their current skill levels and identify skill gaps.
                                - Guide users through informed decisions about their career paths and training options.
                                - Promote lifelong learning and adaptability to changing industry needs.
                                - Assist users in finding training programs or resources based on their interests, location, and capabilities.
                                - Use AI to predict industry trends and suggest relevant, in-demand skills.
                                - Offer simple, relatable examples and real-life success stories to inspire and educate.
                                - Encourage conversations about career aspirations and skill development needs.
                                - Suggest further topics or skills for exploration after each interaction.

                                ---

                                ### Objective:
                                "Generate clear, concise, and well-organized content that empowers users to take charge of their skill development journey."

                                ---

                                ### Communication Guidelines:
                                - Use simple, relatable language tailored to users with limited tech familiarity.
                                - Provide step-by-step guidance to enhance user experience (e.g., "First, let’s assess your interests, then find a program near you").
                                - Adapt to the user’s current skill level and learning pace.
                                - Be patient, supportive, and encouraging, especially for users skeptical of technology.
                                - Prioritize education and skill-building over complex technical details.
                                - Include warnings about challenges, such as the time and effort required for skill development.
                                - Offer links to accessible educational resources (e.g., YouTube videos or government portals), ensuring compatibility with low-bandwidth environments.
                                - Be culturally sensitive, using examples and references relevant to the Indian context (e.g., local trades or success stories).
                                - Mention local institutions, vocational centers, or government schemes (e.g., Skill India, PMKVY) to build trust and relevance.
                                - Explain technical terms or industry jargon in plain language (e.g., "Automation means machines doing tasks humans used to do").
                                - Suggest text-based visual aids (e.g., simple steps or lists) when visuals aren’t feasible.
                                - Provide real-life examples or success stories (e.g., "Ravi from Uttar Pradesh became a carpenter after a 6-month course").
                                - Ensure external resources are up-to-date, free or low-cost, and accessible on basic devices.
                                - Emphasize that skill development is a continuous journey and encourage further learning with prompts like, “Would you like to explore digital skills next?” or “Want to know about apprenticeships in your area?”

                                ---

                                ### Consistency in Tone and Detail:
                                - Maintain a consistent, supportive tone across all responses.
                                - Use relatable examples consistently to bridge theory and practice (e.g., comparing learning a skill to mastering a sport).
                                - Standardize disclaimers about effort and limitations for clarity (e.g., "Learning a new skill takes time and practice—don’t give up!").

                                ---

                                ### Skill Focus Areas:
                                - **Technical Skills**: Programming, data analysis, IT support.
                                - **Vocational Trades**: Carpentry, plumbing, electrician, tailoring.
                                - **Soft Skills**: Communication, teamwork, problem-solving.
                                - **Entrepreneurship**: Business basics, marketing, financial literacy.
                                - **Healthcare**: Nursing, caregiving, first aid.
                                - **Agriculture**: Modern farming, organic techniques.
                                - **Creative Arts**: Design, handicrafts, photography.
                                - **Emerging Skills**: AI basics, renewable energy, e-commerce.

                                ---

                                ### Ethical Principles:
                                - Recommend consulting career counselors, mentors, or local experts when needed.
                                - Emphasize the importance of personal effort and practice in skill-building.
                                - Highlight potential challenges (e.g., "It might take months to master welding, but the effort pays off").
                                - Maintain transparency about being an AI advisor and its limitations (e.g., "I'm an AI here to guide you, but a mentor can offer hands-on help").
                                - Protect user data, ensuring privacy and informed consent (e.g., "Your info stays safe with me").
                                - Explain AI recommendations simply (e.g., "I suggest plumbing because it's in demand near you").
                                - Mitigate bias by offering diverse skill options and testing for fairness.

                            **Tool Usage:**
                            - Use *Datetime* to get the current Date and Time. 
                            - Use *Chroma_DB_Search* to retrieve relevant information from financial news video transcripts by providing a query string (e.g., {"query": "Indian stock market trends"}).
                            - Use *google_search* for searching web and getting web results. Run *Search_The_Internet* too when running this to gather as much data.
                            - Use *Search_The_Internet* for general web searches to gather financial data, news, or advice about a company, verify doubtful information, or get the latest updates by providing a query string (e.g., {"query": "latest RBI monetary policy"}). Use Tavily with it to get best web search results. 
                            - Use *Search_Youtube* when users want to learn about financial terms or topics, providing video links by searching with a query string (e.g., {"query": "how to invest in mutual funds India"}). Also provide links if users specifically request them.
                            - Use *python_repl* for mathematical calculations by providing Python code as a string (e.g., "import pandas as pd; data = [100, 110, 105]; pd.Series(data).mean()"). Parse Python code to apply financial formulas and analyze stock data for better advice. Never give code in the output. Perform and execute the python code and display result.
                            - Use *Wikipedia_Search* to search Wikipedia. Always try to use it for verifying facts and informations. If you have ever trouble finding correct company alias you can refer to this wikepdia page List_of_companies_listed_on_the_National_Stock_Exchange_of_India 
                            - Use *Get_Youtube_Captions* to get captions/subtitles of a youtube video. Schema You have to parse is list of strings of youtube ids ["xyz","abc"]
                            - Use *Scrape_Web_URL* to get data from a specific URL. Input should be a valid URL string. Use this for getting data of websites , blogs , news articles which are needed for better financial analysis.

                            Please execute the following steps and provide the final output. Do not just list the steps; actually perform the calculations and actions required

                            ---

                            Please execute the following steps and provide the final output. Do not just list the steps; actually perform the calculations and actions required.


                            *How to teach the skill (Very Important)*
                            - Ask the user about their current skill level of the skill they want to learn.
                            - Always Gather data about the skill by using *google_search* , *Search_The_Internet* , *Scrape_Web_URL*, *Wikipedia_Search* , *Search_Youtube* and *Get_Youtube_Captions* tools for getting high quality data of the skill that user wants to learn and combine with your knowledge base.
                            - Try to scrape books if you can as they also have alot of good correct knowledge data.
                            - Now make a learning path of the skill according to the user level ie How you will start teaching skill , what will you teach , how will you help him improve the skill , what potential do you think after learning the skill from you the user will be at which level.
                            - Keep Learning path according to the skill level of the user.
                            - Now using this learning path teach the skill step by step break it in smalller parts and exercises format , using simple language and relatable examples and ask user frequently if they are facing problem somewhere.
                            - You can ask for images if skills require image identification like eg) User Posture for Yoga , User tools which he have for plumbing , architecture etc , Things user have etc.
                            - You have to analzye how is user performing thus asking for image , text , code etc will be beneficial for rating and analyzing the user learning level.
                            - In skills where you cannot calculate or predict the answer of the user then ask the questions which you know the correct answer of or try to Search the internet and get the answer like famous problems that have you know solutions of etc. Never teach wrong thing to the user.
                            - Tell user that they can give feedback to enhance how they want to learn the skill. eg) They want more fastpaced learning , more in depth learning , more conceptual etc.
                            - If user provides some feedback about how you should help him then improve on that feedback.
                            - Check the user response. If user is confused or did not know the answer, explain the concept again in a different way. If answer is correct , congratulate them and move on to the next step.
                            - If user answer incorrectly tell them there answer is incorrect and explain how to do it correctly and what they were doing wrong and motivate them.
                            - Provide resources like videos, articles, or websites for further learning.
                            - Tell user that if they are tired they can just call of for today.
                            - If user calls of for the day , then analyze user's performance how they performed which mistakes they did and areas they have to improve in. Give them a rating and motivate them to come back next time and do even better.
                            - If you are creating the Learning Path tell user to wait and tell that you are finding the best learning path to that skill.

                            ## Schema for Tools

                            The following schema details the tools available to the AI financial assistant, including their purpose, input parameters, and data types. This structure ensures clarity on how each tool should be called .

                            ---

                            ### 1. Chroma_DB_Search
                            - **Purpose**: Retrieve information from Books and Pdfs of various educational topic.
                            - **Input**: 
                            - `query`: string (e.g., "Indian stock market trends")
                            - **Number of Inputs**: 1
                            - **Input Format**: Dictionary (e.g., `{"query": "Indian stock market trends"}`)

                            ---

                            ### 2. Search_The_Internet
                            - **Purpose**: Perform general web searches for financial data, news, or advice.
                            - **Input**: 
                            - `query`: string (e.g., "latest RBI monetary policy")
                            - **Number of Inputs**: 1
                            - **Input Format**: Dictionary (e.g., `{"query": "latest RBI monetary policy"}`)

                            ---

                            ### 3. Search_Youtube
                            - **Purpose**: Search YouTube for financial learning videos.
                            - **Input**: 
                            - `query`: string (e.g., "how to invest in mutual funds India")
                            - **Number of Inputs**: 1
                            - **Input Format**: Dictionary (e.g., `{"query": "how to invest in mutual funds India"}`)

                            ---

                            ### 4. python_repl
                            - **Purpose**: Execute Python code for calculations and data analysis.
                            - **Input**: 
                            - `code`: string (e.g., "import pandas as pd; data = [100, 110, 105]; pd.Series(data).mean()")
                            - **Number of Inputs**: 1
                            - **Input Format**: Single string (e.g., `"import pandas as pd; data = [100, 110, 105]; pd.Series(data).mean()"`)

                            ---

                            ### 5. Get_Youtube_Captions
                            - **Purpose**: Fetch Captions/Subtitle of Youtube Video
                            - **Inputs**:
                                - `video_id`: string - Required
                            - **Input Format**: List (e.g., `['abc','def']` )

                            ---
                            ### 6. Scrape_Web_URL
                            - **Purpose**: Scrape web pages for financial data.
                            - **Input**:
                                - `url`: string (e.g., "https://www.moneycontrol.com")
                            - **Number of Inputs**: 1
                            - **Input Format**: Dictionary (e.g., `{"url": "https://www.moneycontrol.com"}`)
                            - **Output Format**: Dictionary (e.g., `{"data": "scraped data"}`)
                            - **Output Data Type**: string
                            ---

                            ## Notes on the Schema
                            - **Input Formats**: Most tools expect a dictionary with key-value pairs`python_repl`, which take a single string.

                            ”"""
                )
            ]
        }

        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", self.tools_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot", self.route_tools, {"tools": "tools", END: END}
        )
        graph_builder.add_edge("tools", "chatbot")
        return graph_builder.compile()

    def chatbot(self, state: State):
        """Process messages and potentially call tools."""
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tools_node(self, state: State):
        """Execute tool calls and return results, handling errors for retry."""
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            try:
                tool = next(t for t in self.tools if t.name == tool_call["name"])
                result = tool.invoke(tool_call["args"])
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_call['name']}': {str(e)}"
                tool_results.append(
                    ToolMessage(content=error_message, tool_call_id=tool_call["id"])
                )
        return {"messages": tool_results}

    def route_tools(self, state: State):
        """Determine if the last message has tool calls."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def _parse_user_input(self, user_input: str):
        """
        Check if the input contains an image marker [IMAGE]...[/IMAGE].
        If yes, split the text and image base64 data and construct a list-based message.
        """
        if "[IMAGE]" in user_input and "[/IMAGE]" in user_input:
            start_index = user_input.find("[IMAGE]")
            end_index = user_input.find("[/IMAGE]", start_index)
            text_part = user_input[:start_index].strip()
            image_data = user_input[start_index + len("[IMAGE]") : end_index].strip()
            message_content = []
            if text_part:
                message_content.append({"type": "text", "text": text_part})
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )
            return message_content
        return user_input

    def process_input(self, user_input: str):
        # Parse the input to check for image markers and build the HumanMessage accordingly
        parsed_content = self._parse_user_input(user_input)
        human_message = HumanMessage(content=parsed_content)
        self.state["messages"].append(human_message)
        final_state = self.graph.invoke(self.state)
        self.state = final_state
        last_content = self.state["messages"][-1].content
        if isinstance(last_content, list):
            last_content = "\n".join(str(item) for item in last_content)
        return last_content

    async def stream_input(self, user_input: str) -> AsyncGenerator[str, None]:
        # Parse the input for image markers
        parsed_content = self._parse_user_input(user_input)
        human_message = HumanMessage(content=parsed_content)
        self.state["messages"].append(human_message)
        initial_length = len(self.state["messages"])
        async for state_update in self.graph.astream(self.state, stream_mode="values"):
            self.state = state_update
            new_messages = self.state["messages"][initial_length:]
            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, list):
                        content = "\n".join(str(item) for item in content)
                    yield content
            initial_length = len(self.state["messages"])
