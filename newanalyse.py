from openai import OpenAI
import sys
import os

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
        openai.api_key = self.api_key
        self.client = openai

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
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class BaseAnalyzer(ABC):
    def __init__(self, file_names: List[str]):
        """
        Abstract base class for all analyzers.
        Shared access to the LLM API is provided via self.llm.
        """
        #self.file_names = file_n
        self.llm = LLM_API()
    def read_file_content(file_path: str) -> str:
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
        self.stop_words = set(stopwords.words('english'))
        
	
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract the most important keywords related to real estate using OpenAI API.
        """
        prompt = f"Extract the top 5 keywords related to real estate from the following text:\n\n{text}\n\nKeywords:"
        try:
            response = self.llm.ask(prompt)
            keywords = response.split(",")  # Assuming the response is a comma-separated list of keywords
            return [keyword.strip() for keyword in keywords]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []
	

		
    def analyze(self) -> Dict:
        """
        Analyze the news articles to extract key information.
        Returns a structured JSON/dictionary output.
        """
        # Step 1: Preprocess the context to extract keywords
        keywords = self.extract_keywords(self.context)
        
        # Step 2: Search through the news articles using the keywords
        #articles = self.search_articles(keywords)
        
        # Step 3: Rate and summarize the articles
        #summaries = self.rate_and_summarize(articles)
        print(keywords)
        return keywords
		
        # use the llm singleton to figure out the 5 most important key words to search for. they shouldnt be too general.

        # use an algorithm to search throuigh "server/data/news" md files with those keywords

        # run each article through an llm to rate it based on how big of an impact on the decision might have. make sure to prompt the ai that it is a underwriter for arch re, etc.
        # summarize the 10 articles with the biggrest impact, return them as a json
if __name__ == "__main__":
    # Example usage
    directory_path =  "./data/news/"
    file_names = BaseAnalyzer.get_all_file_names(directory_path)
    print("what are the file_names?", file_names)
    test_file = file_names[0:1]
    print("test file name is:", test_file)
    # Read the content of all files in the directory and combine them into the context
    context = ""
    for file_name in test_file:
        context += BaseAnalyzer.read_file_content(file_name) + "\n"
    print("context is:", context)
    analyzer = NewsAnalyzer(file_names=test_file, context=context)
    result = analyzer.analyze()
    
    #print("Analysis Result:")
    #print(result)