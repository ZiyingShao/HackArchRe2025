from YoY_analyzer import YoYAnalyzer
from macroeconomics_analyzer import MacroeconomicRiskAnalyzer
from cedant_risk_analyzer import CedantRiskAnalyzer
from news_analyzer import NewsAnalyzer
import os
import json

# Get terms file path from user
terms_path = input("TERMS FILE PATH: ")

# Load JSON and extract specific fields
with open(terms_path, 'r') as f:
    terms_data = json.load(f)

perils = terms_data.get('perils')
currency_code = terms_data.get('currencyCode')
regions = terms_data.get('regions')

# List all files in the same folder with trimmed relative paths
folder_path = os.path.dirname(terms_path)
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
trimmed_files = [path[path.index("server/data/submissions/"):] for path in files if "/server/data/submissions/" in path]
text_files = [f for f in trimmed_files if f.endswith('.md') or f.endswith('.txt')]

economic_files = []
for root, _, files in os.walk("server/data/economics/"):
    for file in files:
        economic_files.append(os.path.join(root, file))

context = f"You are evaluating a company from the following regions: \n {regions}\n They are mainly using {currency_code}."

macroeconomicRiskAnalyzer = MacroeconomicRiskAnalyzer(file_names=economic_files, context=context)
yoyAnalyzer = YoYAnalyzer(file_names=text_files)
cedantRiskAnalyzer = CedantRiskAnalyzer(file_names=trimmed_files)

news_files = []
for root, _, files in os.walk("server/data/news/"):
    for file in files:
        news_files.append(os.path.join(root, file))

newsAnalyzer = NewsAnalyzer(file_names=news_files, context=context)




return_json = {}
return_json["terms"] = terms_data
return_json["yoyAnalysis"] = yoyAnalyzer.analyze()
return_json["macroeconomicAnalysis"] = macroeconomicRiskAnalyzer.analyze()
return_json["cedantRiskAnalysis"] = cedantRiskAnalyzer.analyze()
return_json["newsAnalysis"] = newsAnalyzer.analyze()


with open(folder_path + "/final_json.json", "w") as f:
    json.dump(return_json, f, indent=2)

