import os
import difflib
import json
from base_analyzer import BaseAnalyzer
#Author: Erik Schnell

class YoYAnalyzer(BaseAnalyzer):
    """
    A class to perform Year-over-Year (YoY) analysis on a pair of insurance-related documents.
    The analysis involves comparing the documents for changes and understanding the impact of those changes.
    """

    def analyze(self):
        """
        Analyzes the differences between the provided document pair(s) and returns a detailed list of changes.
        
        The function follows these steps:
        1. Prepares the preview of the files for analysis.
        2. Sends the document previews to an LLM (Language Learning Model) to identify Year-over-Year pairs.
        3. For each identified pair, the function compares the files line-by-line using `difflib`.
        4. Sends the list of changes to the LLM for further analysis.
        5. Returns a dictionary of results, including relevant changes and their impacts.
        
        Returns:
            dict: A dictionary containing either an error message or the results of the comparison.
        """

        # Step 1: Check if there are at least two documents to compare
        if len(self.file_names) < 2:
            return {"error": "Not enough documents for YoY comparison.", "result": None}

        # Step 2: Prepare file previews (first 15 lines) for LLM input
        previews = []
        for path in self.file_names:
            try:
                # Read first 15 lines of the file
                with open(path, "r") as f:
                    lines = "".join([next(f) for _ in range(15)])
                previews.append({"path": path, "preview": lines})
            except Exception as e:
                # Return an error if the file could not be read
                return {"error": f"Failed to read file {path}: {e}", "result": None}

        # Step 3: Construct the LLM prompt to request YoY pairs
        llm_prompt = (
            "You're an underwriting assistant. Based on the preview of these insurance-related documents, "
            "identify valid Year-over-Year (YoY) comparison pairs (e.g. 2023 vs. 2024 contracts). "
            "Only include pairs where both documents cover the same subject and are comparable.\n\n"
            "Return ONLY a **raw JSON array**, where each item is a list of two strings (full paths). "
            "Do NOT format as a code block, do NOT include any explanations or markdown.\n\n"
            "Example format:\n"
            '[\n  ["data/florida/2023_contract.md", "data/florida/2024_contract.md"],\n  ...\n]\n'
            "The JSON should be returned as follows:\n"
            "{\n  \"results\": [[...], [...]]}\n"
        )

        # Add document previews to the LLM prompt
        for p in previews:
            llm_prompt += f"File: {p['path']}\nPreview:\n{p['preview']}\n---\n"

        # Step 4: Execute the LLM prompt and get the response
        try:
            llm_response = self.llm.ask(llm_prompt)
        except Exception as e:
            # Return an error if there was an issue communicating with the LLM
            return {"error": f"Error communicating with LLM: {e}", "result": None}

        # Step 5: Parse the LLM response to extract suggested YoY pairs
        try:
            response_json = json.loads(llm_response)  # Parse JSON response
            suggested_pairs = response_json.get("results", [])  # Extract the "results" key
            print(suggested_pairs)  # Print the suggested pairs for debugging
        except json.JSONDecodeError as e:
            # Return an error if there was an issue parsing the LLM response
            return {"error": f"Error parsing LLM response: {e}", "result": None}

        # Step 6: Return error if no valid YoY pairs were found
        if not suggested_pairs:
            return {"error": "No valid YoY pairs found in LLM response.", "result": None}

        result = {}

        # Step 7: Compare each suggested pair of documents
        for pair in suggested_pairs:
            file1, file2 = pair
            cache_filename = f"cache/{os.path.basename(file1)}_{os.path.basename(file2)}.json"

            # Step 8: Check if results are cached and reuse if available
            if os.path.exists(cache_filename):
                with open(cache_filename, "r") as cache_file:
                    cached_result = json.load(cache_file)
                    result[f"{os.path.basename(file1)} <-> {os.path.basename(file2)}"] = cached_result
                print(f"Using cached result for {os.path.basename(file1)} <-> {os.path.basename(file2)}")
                continue  # Skip the analysis if cached result is found

            # Step 9: Verify the existence of the files before reading them
            if not os.path.exists(file1) or not os.path.exists(file2):
                continue

            with open(file1, "r") as f1, open(file2, "r") as f2:
                content1 = f1.readlines()  # Read content of the first file
                content2 = f2.readlines()  # Read content of the second file

            # Step 10: Use difflib to find the differences between the two documents
            d = difflib.Differ()
            diff = d.compare(content1, content2)
            diff_lines = list(diff)

            changes = []
            for i, line in enumerate(diff_lines):
                # Capture context around each change for better understanding
                context_before = "".join(diff_lines[max(0, i - 3):i])  # 3 lines before
                context_after = "".join(diff_lines[i + 1:i + 4])  # 3 lines after
                change = line.strip()  # Clean up the change

                changes.append({
                    "change": change,
                    "context_before": context_before,
                    "context_after": context_after
                })

            # Step 11: If there are changes, split them into smaller chunks for processing
            if changes:
                change_prompts = []
                for i in range(0, len(changes), 100):  # Break into chunks of 100 changes
                    change_chunk = changes[i:i + 100]
                    change_prompt = (
                        "You are a reinsurance analyst. Given the following changes in the documents,"
                        "first identify if the change is relevant / might have a significant impact."
                        "IGNORE any changes in line breaks, formatting, layout, page numbering or context, page references, format changes. Only analyze significant changes in content.\n"
                        "Then identify the old and new value for RELEVANT changes, explain the impact of the change, "
                        "and provide the context around each RELEVANT change.\n\n"
                        "For each RELEVANT change, please return a JSON object with the following structure:\n"
                        "{\n"
                        "  \"oldValue\": \"value from previous year or null\",\n"
                        "  \"newValue\": \"value from current year or null\",\n"
                        "  \"description\": \"very short description of the impact of the change. e.g. Insolvency or liquidation of the reinsurer triggers contract reassessment or action \",\n"
                        "  \"importance\": \"value from 0.00 to 1.00 on how important this change is\"\n"
                        "}\n\n"
                        "The JSON array of changes should be returned as follows:\n"
                        "{\n  \"results\": [...]}\n"
                    )

                    # Add all changes to the prompt
                    for change in change_chunk:
                        change_prompt += (
                            f"{{\n  \"change\": \"{change['change']}\",\n"
                            f"  \"context_before\": \"{change['context_before']}\",\n"
                            f"  \"context_after\": \"{change['context_after']}\"\n"
                            "}}\n"
                            "---\n"
                        )

                    change_prompts.append(change_prompt)

                # Step 12: Send each chunk to the LLM for further analysis
                all_changes = []
                total_changes = len(change_prompts)
                for idx, change_prompt in enumerate(change_prompts, start=1):
                    llm_response = self.llm.ask(change_prompt)
                    print(f"Processing prompt {idx}/{total_changes}...")

                    if llm_response:
                        try:
                            # Parse the LLM response and extract results
                            response_json = json.loads(llm_response)
                            if 'results' in response_json:
                                all_changes.extend(response_json['results'])
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON response: {e}")

                # Store the results in the final result dictionary
                result[f"{os.path.basename(file1)} <-> {os.path.basename(file2)}"] = all_changes

                # Step 13: Cache the result for future use
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                with open(cache_filename, "w") as cache_file:
                    json.dump(all_changes, cache_file)

        # Step 14: Return error if no meaningful changes are found
        if not result:
            return {"error": "No meaningful YoY changes found.", "result": None}

        # Step 15: Return the final result
        return {"error": None, "result": result}
    
