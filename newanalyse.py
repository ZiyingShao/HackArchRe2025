from openai import OpenAI
import sys
import os
import json 
from collections import OrderedDict
class LLM_API:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLM_API, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.api_key = "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"
        import openai
        self.client = openai
        self.client.api_key = self.api_key

    def ask(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                #response_format={ "type":"json_object" },
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"



from abc import ABC, abstractmethod
from typing import List, Dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class BaseAnalyzer(ABC):
    def __init__(self, file_names: List[str]):
        """
        Abstract base class for all analyzers.
        Shared access to the LLM API is provided via self.llm.
        """
        #self.file_names = file_n
        self.llm = LLM_API()

    def read_file_content(self, file_path: str) -> str:
        """
        Read the content of a file.
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Error: {e}")
            return ""
    @staticmethod
    def get_all_file_names(directory_path: str) -> List[str]:
        """
        Get all file names in a directory.
        """
        try:
            file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
            return file_names
        except Exception as e:
            print(f"Error: {e}")
            return []

    @abstractmethod
    def analyze(self) -> Dict:
        """
        Each analyzer must implement this method.
        Returns a structured JSON/dictionary output.
        """




from typing import Dict, List

class NewsAnalyzer(BaseAnalyzer):
    def __init__(self, file_names: List[str], context: str):
        super().__init__(file_names)

        self.context = context  # Context to guide the AI
        self.file_names = file_names
        self.llm = LLM_API()
        self.matched_articles = []  # Access the shared LLM API instance


    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract the most important keywords related to real estate using OpenAI API.
        """
        prompt = f"Extract the top 10 keywords related to real estate from the following text:\n\n{text}\n\nKeywords:"
        try:
            response = self.llm.ask(prompt)
            # Split the response by newlines and remove numbering
            keywords = [line.split('. ', 1)[-1].strip() for line in response.split('\n') if line.strip()]
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    
    def search_articles(self, keywords: List[str], auto_extract= False) -> List[Dict]:
        """
        Search through the articles for the given keywords.
        Returns a list of dictionaries containing the article title, content, and matched keywords.
        """
        matching_articles = []
        for file_name in self.file_names:
            content = self.read_file_content(file_name)

            if auto_extract:
                extracted_keywords = self.extract_keywords(content)
                matched = extracted_keywords

            # Find which keywords actually match
            else:
                matched = [kw for kw in keywords if kw.lower() in content.lower()]

            if matched:
                title = os.path.basename(file_name).replace('_', ' ').replace('.md', '').title()

                matching_articles.append({
					"File Name": file_name,
                    "Title": title,
                    "Content": content,
                    "Matched Keywords": matched  # Add matched keywords here!
                })
        return matching_articles





    def analyze(self) -> Dict:
        """
        Analyze the news articles to extract key information.
        Returns a structured JSON/dictionary output.
        """
        # Step 1: Preprocess the context to extract keywords
        keywords = self.extract_keywords(self.context)
        print("Extracted Keywords:", keywords)


        # Step 2: Search through the news articles using the keywords
        articles = self.search_articles(keywords)
        print("Found Articles:", articles)
        #articles = self.search_articles(keywords)

        # Step 3: Rate and summarize the articles
        summaries = []
        risk_level_mapping = {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Unknown": -1
        }
        for article in self.matched_articles:
            prompt = f"""
            You are an expert underwritting analyst.
    
            Given the following news article, do the following:
            
            1. Summarize the article in 3-5 sentences.
            2. Extract 3-5 important points or key facts from the article.
			2. Identify any potential risks mentioned or implied in the article.
	        3. Assign an overall Risk Level to the article based on the potential impact (choose one: High, Medium, Low).
			4. Assign an overall Risk Level (High, Medium, Low).
			5. Based on the risk level and risks, provide a short Recommendation for Underwriters (e.g., "Proceed with caution", "Increase premium", "Exclude certain coverages", etc.)
    
            Format your response strictly in this JSON format:
    
            {{
             "Summary": "<summary>",
             "Important Points": ["point 1", "point 2", "point 3"],
			 "Potential Risks": ["risk 1", "risk 2", "risk 3"],
             "Risk Level": "High/Medium/Low",
			 "Recommendation for Underwriters": "<recommendation>"
            }}
    
            Here is the article:
            \"\"\"
            {article['Content']}
            \"\"\"
            """
    
            response = self.llm.ask(prompt)
    
            try:
                # Try to parse response as JSON
                parsed = json.loads(response)
    
                # Build a clean ordered dict
                article_summary = OrderedDict()
                article_summary["Title"] = article["Title"]
                article_summary["Summary"] = parsed.get("Summary", "").strip()
                article_summary["Important Points"] = parsed.get("Important Points", [])
                article_summary["Potential Risks"] = parsed.get("Potential Risks", [])
                risk_level = parsed.get("Risk Level", "").strip()
                if risk_level not in ["High", "Medium", "Low"]:
                    risk_level = "Unknown"
                risk_level_numeric = risk_level_mapping[risk_level]
                article_summary["Risk Level"] = risk_level
                article_summary["Risk Level Numeric"] = risk_level_numeric
                article_summary["Recommendation"] = parsed.get("Recommendation for Underwriters", "").strip()

                article_summary["Keywords"] = article.get("Matched Keywords", [])
    
            except json.JSONDecodeError:
                # If it's not valid JSON, keep the text as-is and leave points empty
                print(f"Failed to decode JSON for article: {article['Title']}")
                article_summary = OrderedDict()
                article_summary["Title"] = article["Title"]
                article_summary["Summary"] = response.strip()
                article_summary["Keywords"] = article.get("Matched Keywords", [])
                article_summary["Important Points"] = []
                article_summary["Potential Risks"] = []
                article_summary["Risk Level"] = "Unknown"
                article_summary["Recommendation for Underwriters"] = "No recommendation due to JSON error."

                
                #print(f"Failed to decode JSON for article: {article['Title']}")
                #print(f"Response: {response}")
            summaries.append(article_summary)

        print("the summary is:", summaries)
        return summaries


        # use the llm singleton to figure out the 5 most important key words to search for. they shouldnt be too general.

        # use an algorithm to search throuigh "server/data/news" md files with those keywords

        # run each article through an llm to rate it based on how big of an impact on the decision might have. make sure to prompt the ai that it is a underwriter for arch re, etc.
        # summarize the 10 articles with the biggrest impact, return them as a json
if __name__ == "__main__":
    # Example usage
    directory_path =  "./data/news/"
    file_names = BaseAnalyzer.get_all_file_names(directory_path)
    print("what are the file_names?", file_names)
    test_file = file_names[0:10]
    print("test file name is:", test_file)
    # Read the content of all files in the directory and combine them into the context
    analyzer = NewsAnalyzer(file_names=test_file, context="")

    use_predefined_keywords = True
    given_keywords = ['catastrophe', 'catastrophic', 'florida', 'location', 'turkey', 'inflation', 'earthquake', 'hurricane', 'flood', 'storm','insurance', 'reinsurance', 'property', 'real estate', 'market', 'investment', 'risk', 'loss', 'damage', 'claim']
    
    if use_predefined_keywords:
        articles = analyzer.search_articles(given_keywords, auto_extract=False)
    else:
        articles = analyzer.search_articles([], auto_extract=True) 
    print(f"Found {len(articles)} matching articles.")
    analyzer.matched_articles = articles
    analyzer.file_names = [a["File Name"] for a in articles]
    analyzer.context = ""
    analyzer.articles = articles

    result = analyzer.analyze()
	# Save the result to a JSON file
    output_file_path = "/Users/ziyingshao/Desktop/HackArchRe2025-main/analysis_result.json"
    with open(output_file_path, "w") as json_file:
        json.dump(result, json_file, indent=4)
    #print("Analysis Result:")
    #print(result)