import os
import json
from typing import List, Dict
from LLM_API import LLM_API
from base_analyzer import BaseAnalyzer

CACHE_DIR = "/cache/"

class NewsAnalyzer(BaseAnalyzer):
    def __init__(self, file_names: List[str], context: str):
        super().__init__(file_names)
        self.context = context
        self.file_names = file_names
        self.llm = LLM_API()  # Shared LLM API instance

    def _get_cache_file_path(self) -> str:
        """Get the cache file path based on the context."""
        # Use a simple file name derived from the context
        cache_filename = f"{self.context.replace(' ', '_')}.json"
        return os.path.join(CACHE_DIR, cache_filename)

    def _load_from_cache(self) -> Dict:
        """Load the cached analysis result if available."""
        cache_path = self._get_cache_file_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, data: Dict):
        """Save the analysis result to cache."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = self._get_cache_file_path()
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords related to the context using LLM API, expecting a JSON response."""
        prompt = f"""
        Extract the top 10 keywords related to the context from the following text and return the response as a JSON list.
        Context: {self.context}
        
        Text:
        {text}
        
        Expected JSON format:
        {{
            "keywords": ["keyword1", "keyword2", "keyword3", ..., "keyword10"]
        }}
        """
        
        try:
            # Ask the LLM for keywords in JSON format
            response = self.llm.ask(prompt)
            
            # Try to parse the response as JSON
            parsed_response = json.loads(response)

            # Extract the list of keywords from the parsed response
            keywords = parsed_response.get("keywords", [])
            return [keyword.strip() for keyword in keywords if keyword.strip()]
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error extracting keywords: {e}")
            return []
        
    def search_articles(self, keywords: List[str]) -> List[Dict]:
        """Search articles for keywords and return matched articles."""
        matching_articles = []
        for file_name in self.file_names:
            content = self.read_markdown_file(file_name)
            matched = [kw for kw in keywords if kw.lower() in content.lower()]
            if matched:
                title = os.path.basename(file_name).replace('_', ' ').replace('.md', '').title()
                matching_articles.append({
                    "Title": title,
                    "Content": content,
                    "Matched Keywords": matched
                })
        return matching_articles

    def read_markdown_file(self, file_name: str) -> str:
        """Read the content of a markdown (.md) file."""
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            return ""

    def analyze(self) -> Dict:
        """Analyze news articles, extract key information, and return a structured output."""
        # Check if the result is already cached
        cached_data = self._load_from_cache()
        if cached_data:
            print("Cache hit - returning cached result.")
            return cached_data

        # Step 1: Extract keywords from context
        keywords = self.extract_keywords(self.context)
        print("Extracted Keywords:", keywords)

        # Step 2: Search articles using keywords
        articles = self.search_articles(keywords)
        print("Found Articles:", articles)

        # Step 3: Analyze articles
        summaries = []
        risk_level_mapping = {"Low": 0, "Medium": 1, "High": 2, "Unknown": -1}

        for article in articles:
            prompt = f"""
            You are an expert underwriting analyst.

            Given the following news article, do the following:
            
            1. Summarize the article in 3-5 sentences.
            2. Extract 3-5 important points or key facts from the article.
            3. Identify any potential risks mentioned or implied in the article.
            4. Assign an overall Risk Level to the article based on the potential impact (choose one: High, Medium, Low).

            Format your response strictly in this JSON format:

            {{
                "Summary": "<summary>",
                "Important Points": ["point 1", "point 2", "point 3"],
                "Potential Risks": ["risk 1", "risk 2", "risk 3"],
                "Risk Level": "<Risk Level>"
            }}

            Here is the article:
            \"\"\"
            {article['Content']}
            \"\"\"
            """
            response = self.llm.ask(prompt)

            try:
                # Parse response as JSON
                parsed = json.loads(response)
                article_summary = {
                    "Title": article["Title"],
                    "Summary": parsed.get("Summary", "").strip(),
                    "Important Points": parsed.get("Important Points", []),
                    "Potential Risks": parsed.get("Potential Risks", []),
                    "Risk Level": parsed.get("Risk Level", "Unknown")
                }

                # Convert risk level to numeric
                risk_level = parsed.get("Risk Level", "Unknown").strip()
                article_summary["Risk Level Numeric"] = risk_level_mapping.get(risk_level, -1)
                article_summary["Keywords"] = article.get("Matched Keywords", [])

            except json.JSONDecodeError:
                print(f"Failed to decode JSON for article: {article['Title']}")
                article_summary = {
                    "Title": article["Title"],
                    "Summary": response.strip(),
                    "Keywords": article.get("Matched Keywords", []),
                    "Important Points": [],
                    "Potential Risks": [],
                    "Risk Level": "Unknown"
                }

            summaries.append(article_summary)

        # Save to cache for future use
        self._save_to_cache({"Articles": summaries})
        print("Summary saved to cache.")
        
        return {"Articles": summaries}