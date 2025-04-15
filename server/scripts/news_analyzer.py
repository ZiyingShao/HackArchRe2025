import spacy
from openai import OpenAI
import sys
import os
import json
from collections import OrderedDict
import re
from collections import Counter
import matplotlib.pyplot as plt

#Author: Ziying Shao
class LLM_API:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLM_API, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.api_key = "put your key here"
        import openai

        self.client = openai
        self.client.api_key = self.api_key

    def ask(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                # response_format={ "type":"json_object" },
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"


from abc import ABC, abstractmethod
from typing import List, Dict


class BaseAnalyzer(ABC):
    def __init__(self, file_names: List[str]):
        """
        Abstract base class for all analyzers.
        Shared access to the LLM API is provided via self.llm.
        """
        # self.file_names = file_n
        self.llm = LLM_API()

    @staticmethod
    def smart_json_parse(response_text):
        """
        Tries to parse the LLM output more robustly, including nested JSONs and fixing missing commas.
        """
        try:
            # Try normal JSON parse first
            parsed = json.loads(response_text)

            # If parsed successfully but "Summary" accidentally contains another JSON blob inside a string
            if isinstance(parsed.get("Summary", ""), str) and parsed[
                "Summary"
            ].strip().startswith("{"):
                try:
                    nested = json.loads(parsed["Summary"])
                    parsed = nested
                except json.JSONDecodeError:
                    print("Warning: Nested JSON in 'Summary' field couldn't be parsed.")

            return parsed

        except json.JSONDecodeError:
            # Attempt quick fixes
            fixed_text = (
                response_text.replace("\n", "")
                .replace("\t", "")
                .replace("\\n", "")
                .replace("\\t", "")
            )

            # Add missing commas between JSON fields (common LLM mistake)
            fixed_text = re.sub(r"\"\s*\"", '", "', fixed_text)

            # Try parsing again
            try:
                parsed = json.loads(fixed_text)
                return parsed
            except:
                return None

    def read_file_content(self, file_path: str) -> str:
        """
        Read the content of a file.
        """
        try:
            with open(file_path, "r") as file:
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
            file_names = [
                os.path.join(directory_path, f)
                for f in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, f))
            ]
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

    def extract_top_locations(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract top N most frequent locations from the given text using spaCy.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        location_counts = Counter(locations)
        top_locations = [loc for loc, count in location_counts.most_common(top_n)]
        return top_locations

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract the most important keywords related to real estate using OpenAI API.
        """
        prompt = f"Extract the top 10 keywords related to real estate from the following text:\n\n{text}\n\nKeywords:"
        try:
            response = self.llm.ask(prompt)
            # Split the response by newlines and remove numbering
            keywords = [
                line.split(". ", 1)[-1].strip()
                for line in response.split("\n")
                if line.strip()
            ]
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def search_articles(self, keywords: List[str], auto_extract=False) -> List[Dict]:
        matching_articles = []
        nlp = spacy.load("en_core_web_sm")  # load only once
        for file_name in self.file_names:
            content = self.read_file_content(file_name)

            # Extract top locations
            doc = nlp(content)
            locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            location_counts = Counter(locations)
            top_locations = [loc for loc, count in location_counts.most_common(5)]

            matched_locations = [
                loc
                for loc in top_locations
                if any(kw.lower() in loc.lower() for kw in keywords)
            ]

            if matched_locations:
                title = (
                    os.path.basename(file_name)
                    .replace("_", " ")
                    .replace(".md", "")
                    .title()
                )

                matching_articles.append(
                    {
                        "File Name": file_name,
                        "Title": title,
                        "Content": content,
                        "Top Locations": top_locations,
                        "Matched Locations": matched_locations,
                        "Matched Keywords": matched_locations,  # reuse for downstream
                    }
                )
        return matching_articles

    def analyze(self) -> Dict:
        """
        Analyze the news articles to extract key information.
        Returns a structured JSON/dictionary output.
        """
        # Step 1: Preprocess the context to extract keywords
        # keywords = self.extract_keywords(self.context)
        # print("Extracted Keywords:", keywords)

        # Step 2: Search through the news articles using the keywords
        # articles = self.search_articles(keywords)
        # print("Found Articles:", articles)
        # articles = self.search_articles(keywords)

        # Step 3: Rate and summarize the articles
        summaries = []

        for article in self.matched_articles:
            prompt = f"""
            You are an expert underwriting analyst for a major reinsurance company
    
            Given the following news article, please provide a deep insurance analysis focusing on underwriting impacts:
            
			Tasks:
			1. **Generate a Professional Synthetic Title** for this article (short, clear, 5-8 words).
            2. Summarize the article in **one sentence (within 10 words)**, clearly mentioning any geographic location (e.g., 'Zurich')
        	3. Identify the **Nature of the Risk** in one sentence.
			4. List the **Core Risks** (3-5 types of risks) using short bullet points (e.g., 'Governance Risk', 'Operational Risk', 'Financial Risk')
			5. List the **Risk Factors** (specific causes contributing to the risk) in bullet points.
			6. **Risk Events**: List 2-5 specific events that could happen based on the article (e.g., 'Board deadlock', 'CEO replacement', 'Credit rating downgrade'), assign its own Impact Score (1–3) and Likelihood Score (1–3) separately.
			
			9. **Suggestion for Risk Mitigation**: Give 1-2 sentences suggesting practical ways to reduce or control the risk.
			10. **Premium Outlook**: Recommend whether the insurance premium should increase, decrease, or stay stable. If adjusting, suggest by how much (e.g., '+20% premium') and explain why briefly.
			11. Based on the risk level and risks, provide a short Recommendation for Underwriters (e.g. "Increase premium", "Exclude certain coverages", etc.)
			12. Provide a **Suggestion for Risk Mitigation** in 1-2 sentences, focusing on practical measures.
		    

			IMPORTANT:
			- Your summary must be concise and relevant to the article.
			- Your answers must be based ONLY on facts clearly stated or implied in the article.
            - DO NOT invent risks that are not related to the article.
			- Your recommendation must be logically based on the extracted Important Points and Potential Risks.
            - DO NOT invent new risks not mentioned.
            - Justify your recommendation briefly based on the risks identified
            - Be professional, concise, and insurance-focused.
            - Return the answer in the following strict JSON format:
    
            {{
			 "Synthetic Title": "<new title>",
             "Summary": "<summary>",
			 "Important Points": ["point 1", "point 2", "point 3"],
			 "Nature of Risk": "<sentence>",
			 "Risk Events": [{{
								"Event": "<event1>",
								"Impact": <1/2/3>,
								"Likelihood": <1/2/3>
								}},
								{{
								"Event": "<event2>",
								"Impact": <1/2/3>,
								"Likelihood": <1/2/3>
								}}
							],
			 "Core Risks": ["<risk1>", "<risk2>", "<risk3>"],
			 "Risk Factors": ["<factor1>", "<factor2>", "<factor3>"],
			 "Average Impact Score": <1/2/3>,
			 "Average Likelihood Score": <1/2/3>,
			 "Suggestion for Risk Mitigation": "<short mitigation strategy>",
			 "Premium Outlook": "<premium adjustment and brief justification>",
			 "Recommendation for Underwriters": "<recommendation>"
            }}
    
            Here is the article:
            \"\"\"
            {article['Content']}
            \"\"\"
            """

            response = self.llm.ask(prompt)
            article_summary = OrderedDict()
            parsed = BaseAnalyzer.smart_json_parse(response)

            if parsed:
                # Base information
                article_summary["Synthetic Title"] = parsed.get(
                    "Synthetic Title", ""
                ).strip()
                article_summary["Summary"] = parsed.get("Summary", "").strip()
                article_summary["Source Article Index"] = article["Title"]
                article_summary["Nature of Risk"] = parsed.get(
                    "Nature of Risk", ""
                ).strip()
                article_summary["Core Risks"] = parsed.get("Core Risks", [])
                article_summary["Risk Factors"] = parsed.get("Risk Factors", [])
                article_summary["Suggestion for Risk Mitigation"] = parsed.get(
                    "Suggestion for Risk Mitigation", ""
                ).strip()
                article_summary["Premium Outlook"] = parsed.get(
                    "Premium Outlook", ""
                ).strip()
                article_summary["Top Locations"] = article.get("Top Locations", [])
                article_summary["Keywords"] = article.get("Matched Keywords", [])

                # Risk Event Information
                risk_events = parsed.get("Risk Events", [])
                risk_matrix_data = []
                impact_list = []
                likelihood_list = []

                for event_info in risk_events:
                    if isinstance(event_info, dict):
                        event_name = event_info.get("Event", "")
                        impact = event_info.get("Impact", 1)
                        likelihood = event_info.get("Likelihood", 1)
                    else:
                        event_name = event_info
                        impact = parsed.get("Impact Score", 1)
                        likelihood = parsed.get("Likelihood Score", 1)

                    risk_matrix_data.append(
                        {
                            "Event": event_name,
                            "Impact": impact,
                            "Likelihood": likelihood,
                        }
                    )
                    impact_list.append(impact)
                    likelihood_list.append(likelihood)

                article_summary["Risk Matrix Data"] = risk_matrix_data

                # Average Scores
                if impact_list:
                    article_summary["Average Impact Score"] = sum(impact_list) / len(
                        impact_list
                    )
                else:
                    article_summary["Average Impact Score"] = parsed.get(
                        "Impact Score", 0
                    )

                if likelihood_list:
                    article_summary["Average Likelihood Score"] = sum(
                        likelihood_list
                    ) / len(likelihood_list)
                else:
                    article_summary["Average Likelihood Score"] = parsed.get(
                        "Likelihood Score", 0
                    )

                article_summary["Risk Priority Number"] = (
                    article_summary["Average Impact Score"]
                    * article_summary["Average Likelihood Score"]
                )

                # Plot the Risk Matrix
                if risk_matrix_data:
                    fig, ax = plt.subplots()

                    for risk in risk_matrix_data:
                        ax.scatter(
                            risk["Likelihood"],
                            risk["Impact"],
                            s=100,
                            marker="x",
                            label=risk["Event"],
                        )
                        ax.text(
                            risk["Likelihood"] + 0.05,
                            risk["Impact"] + 0.05,
                            risk["Event"],
                            fontsize=8,
                        )

                    ax.set_xlim(0.5, 3.5)
                    ax.set_ylim(0.5, 3.5)
                    ax.set_xlabel("Likelihood (1 = Low, 3 = High)")
                    ax.set_ylabel("Impact (1 = Low, 3 = High)")
                    ax.set_title(f"Governance Risk Matrix for {article['Title']}")
                    ax.grid(True)
                    plt.legend(loc="best", fontsize="small")

                    output_dir = "./output/"
                    os.makedirs(output_dir, exist_ok=True)
                    sanitized_title = (
                        article["Title"].replace(" ", "_").replace("/", "_")
                    )
                    image_filename = f"{output_dir}matrix_{sanitized_title}.png"
                    plt.savefig(image_filename, bbox_inches="tight")
                    plt.close()

                    article_summary["Risk Matrix Image"] = image_filename

            else:
                print(f"Failed to decode JSON for article: {article['Title']}")
                article_summary = OrderedDict()
                article_summary["Source Article Index"] = article["Title"]
                article_summary["Summary"] = response.strip()
                article_summary["Keywords"] = article.get("Matched Keywords", [])
                article_summary["Important Points"] = []
                article_summary["Potential Risks"] = []
                article_summary["Risk Level"] = "Unknown"
                article_summary["Recommendation for Underwriters"] = (
                    "No recommendation due to JSON error."
                )
                article_summary["Risk Priority Number"] = 0

            summaries.append(article_summary)

            print("the summary is:", summaries)
        return summaries

        # use the llm singleton to figure out the 5 most important key words to search for. they shouldnt be too general.

        # use an algorithm to search throuigh "server/data/news" md files with those keywords

        # run each article through an llm to rate it based on how big of an impact on the decision might have. make sure to prompt the ai that it is a underwriter for arch re, etc.
        # summarize the 10 articles with the biggrest impact, return them as a json


if __name__ == "__main__":
    # Example usage
    directory_path = "./data/news/"
    file_names = BaseAnalyzer.get_all_file_names(directory_path)

    test_file = file_names[0:150]

    # Read the content of all files in the directory and combine them into the context
    analyzer = NewsAnalyzer(file_names=test_file, context="")
    print("Extracting top locations for each article...")
    for file_path in test_file:
        content = analyzer.read_file_content(file_path)
        top_locations = analyzer.extract_top_locations(
            content
        )  # <- assuming you have this method
        print(f"File: {os.path.basename(file_path)} | Top Locations: {top_locations}")
    use_predefined_keywords = True
    given_keywords_location = ["florida", "california"]

    if use_predefined_keywords:
        articles = analyzer.search_articles(given_keywords_location, auto_extract=False)
    else:
        articles = analyzer.search_articles([], auto_extract=True)
    print(f"Found {len(articles)} matching articles.")
    analyzer.matched_articles = articles
    analyzer.file_names = [a["File Name"] for a in articles]
    analyzer.context = ""
    analyzer.articles = articles

    result = analyzer.analyze()
    # Save the result to a JSON file with wrapped metadata

    output_file_path = (
        "/Users/ziyingshao/Desktop/HackArchRe2025-main/analysis_result.json"
    )
    with open(output_file_path, "w") as json_file:
        json.dump(result, json_file, indent=4)

    print(f"Analysis completed. {len(result)} articles saved.")

    # Save the result to a JSON file
    output_file_path = (
        "/Users/ziyingshao/Desktop/HackArchRe2025-main/analysis_result.json"
    )
    with open(output_file_path, "w") as json_file:
        json.dump(result, json_file, indent=4)
    # print("Analysis Result:")
    # print(result)
