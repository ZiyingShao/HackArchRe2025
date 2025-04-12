from typing import Dict, List
from base_analyzer import BaseAnalyzer
import os
import json

class CedantRiskAnalyzer(BaseAnalyzer):
    def __init__(self, file_names: List[str]):
        super().__init__(file_names)

    def analyze(self) -> Dict:
        output_dict = {}
        self.excel_to_pdf()

        # Get all PDF files
        pdf_files = [f for f in self.file_names if f.lower().endswith(".pdf")]

        for file_path in pdf_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_path = os.path.join("cache", f"{base_name}.json")

            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    response_data = json.load(f)
                output_dict[base_name] = response_data
                continue

            llm_prompt = """
You are an underwriter at Arch Re analyzing a PDF document. Follow these steps:

1. Understand the overall content and decode any reinsurance-specific acronyms.
2. Identify key numerical insights (totals, averages, trends).
3. Determine which values are worth comparing (e.g., over time, across companies, etc.).
4. Perform the comparisons using actual numbers and assess their impact on risk or investment potential.
5. Provide your findings in raw JSON only (no code block, no markdown, no explanations) using the format below:

{
  "response": [
    {
      "data_point": "",
      "comparison": "",
      "possible_risk": "",
      "risk_score": 0.6
    }
  ]
}

Make sure:
- The output is valid JSON
- Keys and string values are double-quoted
- The risk_score ranges from -1 (no risk) to 1 (high risk)
"""

            llm_response = self.llm.ask_with_pdf(prompt=llm_prompt, pdf_path=file_path)

            try:
                response_data = json.loads(llm_response)
            except json.JSONDecodeError:
                print(f"Invalid JSON in response for file: {file_path}")
                continue

            # Save to cache
            with open(json_path, "w") as f:
                json.dump(response_data, f, indent=2)

            output_dict[base_name] = response_data

        return output_dict
    
    def excel_to_pdf(self):
        # Not yet implemented
        self.file_names = self.file_names
        return
