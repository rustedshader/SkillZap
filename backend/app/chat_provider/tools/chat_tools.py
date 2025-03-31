from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import BraveSearch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import YoutubeLoader


class ChatTools:
    def __init__(
        self,
        duckduckgo_general: DuckDuckGoSearchResults = None,
        duckduckgo_news: DuckDuckGoSearchResults = None,
        searxng: SearxSearchWrapper = None,
        brave_search: BraveSearch = None,
        youtube_search: YouTubeSearchTool = None,
    ):
        self.duckduckgo_general = duckduckgo_general
        self.duckduckgo_news = duckduckgo_news
        self.searxng = searxng
        self.brave_search = brave_search
        self.youtube_search = youtube_search
        self.youtube_captioner = YouTubeSearchTool

        # Initialize mftool for mutual funds data

    def get_youtube_captions(self, video_ids: list[str]):
        """Retrieve captions and video info for a list of YouTube video IDs."""
        if not self.youtube_captioner:
            return "YouTube captioner tool not initialized."

        if not video_ids:
            return "No video IDs provided."

        # Update video_ids in the youtube_captioner instance
        self.youtube_captioner.video_ids = video_ids

        try:
            results = []
            for video_id in video_ids:
                loader = YoutubeLoader.from_youtube_url(
                    f"https://www.youtube.com/watch?v={video_id}",
                    add_video_info=False,
                    language=["en"],
                )
                video_data = loader.load()
                if video_data:
                    results.extend(video_data)

            if results:
                return "\n".join([str(result) for result in results])
            return "No captions or data found for the provided video IDs."

        except Exception as e:
            return f"Error retrieving YouTube captions: {str(e)}"

    def search_wikipedia(self, query: str):
        """Search Wikipedia and return the content of the first matching page."""
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
        return docs[0].page_content if docs else "No results found."

    def search_searxng(self, query: str):
        """Search using SearxNG and return the results."""
        if self.searxng:
            return self.searxng.run(query)
        return "SearxNG search tool not initialized."

    def search_brave(self, query: str):
        """Search using Brave Search and return the results."""
        if self.brave_search:
            return self.brave_search.run(query)
        return "Brave Search tool not initialized."

    def search_youtube(self, query: str):
        """Search YouTube and return results as a formatted string."""
        if self.youtube_search:
            results = self.youtube_search.run(query)
            if isinstance(results, list):
                return "\n".join([str(result) for result in results])
            return str(results)
        return "Youtube Search tool not initialized"

    def search_duckduckgo(
        self, query: str, backend: str = "general", output_format: str = None
    ):
        """
        Search using DuckDuckGo with specified backend and output format.
        Args:
            query (str): The search query.
            backend (str): "general" or "news" to select the search backend (default: "general").
            output_format (str): "list" to return results as a list, None for a string (default: None).
        Returns:
            Results as a list or formatted string based on output_format.
        """
        if backend == "general" and self.duckduckgo_general:
            search_tool = self.duckduckgo_general
        elif backend == "news" and self.duckduckgo_news:
            search_tool = self.duckduckgo_news
        else:
            return f"Invalid backend: {backend} or tool not initialized."

        results = search_tool.invoke(query)
        if output_format == "list":
            return results
        else:
            if isinstance(results, list):
                return "\n".join([str(result) for result in results])
            return str(results)

    def search_web(self, query: str, max_results: int = 5):
        """
        Search the web using multiple search engines and combine results.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return from each search engine (default: 5).

        Returns:
            str: Combined search results from different search engines.
        """
        # Initialize results list
        combined_results = []

        # Search SearxNG
        try:
            if self.searxng:
                searxng_results = self.searxng.run(query)
                if isinstance(searxng_results, list):
                    combined_results.extend(searxng_results)
                else:
                    combined_results.append(searxng_results)
        except Exception as e:
            combined_results.append(f"SearxNG Search Error: {str(e)}")

        # Search Brave
        try:
            if self.brave_search:
                brave_results = self.brave_search.run(query)
                if isinstance(brave_results, list):
                    combined_results.extend(brave_results)
                else:
                    combined_results.append(brave_results)
        except Exception as e:
            combined_results.append(f"Brave Search Error: {str(e)}")

        # Search DuckDuckGo (general)
        try:
            if self.duckduckgo_general:
                ddg_results = self.search_duckduckgo(query)
                if isinstance(ddg_results, list):
                    combined_results.extend(ddg_results)
                else:
                    combined_results.append(ddg_results)
        except Exception as e:
            combined_results.append(f"DuckDuckGo Search Error: {str(e)}")

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in combined_results:
            result_str = str(result)
            if result_str not in seen:
                seen.add(result_str)
                unique_results.append(result)

        # Return results as a formatted string
        if unique_results:
            return "\n\n".join([str(result) for result in unique_results])
        else:
            return "No search results found."

    def scrape_web_url(self, url: str) -> str:
        """
        Scrape content from a given URL.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: The scraped content or error message.
        """
        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                return "Error: Invalid URL. Must start with http:// or https://"

            # Load and parse the URL
            loader = WebBaseLoader(url)
            docs = loader.load()

            # Check if any content was retrieved
            if not docs:
                return "Error: No content found at the URL"

            return docs[0].page_content

        except ValueError as e:
            return f"URL Error: {str(e)}"
        except Exception as e:
            return f"Error scraping URL {url}: {str(e)}"
