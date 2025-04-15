import json
import os
from typing import Dict, List
from base_analyzer import BaseAnalyzer
import pandas as pd
#Author: Erik Schnell

class MacroeconomicRiskAnalyzer(BaseAnalyzer):
    """
    Analyzer class for analyzing macroeconomic risks such as FX rates, inflation rates, and interest rates.
    
    This class loads data files, performs calculations, and saves the results in a cache to avoid redundant processing.
    It focuses on the specified currencies and returns only the relevant data for those currencies.
    """
    
    def __init__(self, file_names: List[str], context: str):
        """
        Initializes the analyzer with the provided files and context.
        
        Args:
            file_names (List[str]): List of file paths to the data files.
            context (str): Context to guide the AI in selecting relevant files and currencies.
        """
        super().__init__(file_names)
        self.context = context  # Context to guide the AI
    
    def analyze(self) -> Dict:
        """
        Analyzes the macroeconomic data based on the provided files.
        
        It processes FX rates, inflation rates, and interest rates. The analysis results are returned in a dictionary.
        
        Returns:
            Dict: Dictionary containing the analysis results for FX rates, inflation rates, and interest rates.
        """
        results = {
            "fx_rates": None,
            "inflation_rates": None,
            "interest_rates": None
        }

        # use AI to find relevant file names and currencies based on context
        llm_prompt = (
            f"You're an underwriting assistant analyzing insurance based on the following context: {self.context}. "
            "Find any relevant files and currencie codes for this analysis. Better give too many than too little. Try to include at least one fx-rates, one interest-rates and one inflation-rates file.\n"
            "If provided with a country, make sure to include all the currency codes used in this country in the response.\n"
            "Return ONLY a **raw JSON array**, where each item is a list of two strings (full paths). "
            "Do NOT format as a code block, do NOT include any explanations or markdown.\n\n"
            "Expected format:\n"
            "{\n  \"relevant_file_names\": ['/server/.../file1.csv', '/server/.../file2.csv'], \"relevant_currency_codes\": ['USD']}\n"
            f"File Names: \n {self.file_names}"
        )

        # Step 4: Execute the LLM prompt and get the response
        try:
            llm_response = self.llm.ask(llm_prompt)
        except Exception as e:
            # Return an error if there was an issue communicating with the LLM
            return {"error": f"Error communicating with LLM: {e}", "result": None}

        # Step 5: Parse the LLM response to extract suggested files and currencies
        try:
            print(llm_response)
            response_json = json.loads(llm_response)  # Parse JSON response
            relevant_file_names = response_json.get("relevant_file_names", [])
            relevant_currency_codes = response_json.get("relevant_currency_codes", [])
            # Filter only the relevant files and currencies
            self.file_names = relevant_file_names
            self.currencies = relevant_currency_codes  # Assign the relevant currencies from the AI response

        except json.JSONDecodeError as e:
            # Return an error if there was an issue parsing the LLM response
            return {"error": f"Error parsing LLM response: {e}", "result": None}

        # Now only analyze the relevant files
        fx_rate_files = [file for file in self.file_names if "fx-rates" in file]
        inflation_rate_files = [file for file in self.file_names if "inflation-rate" in file]
        interest_rate_files = [file for file in self.file_names if "interest-rates" in file]
        print(self.file_names)
        print(self.currencies)
        print(f"interest rate files {interest_rate_files}")

        if fx_rate_files:
            results["fx_rates"] = self.analyze_fx_rates(fx_rate_files)

        if inflation_rate_files:
            results["inflation_rates"] = self.analyze_inflation_rates(inflation_rate_files)

        if interest_rate_files:
            print("analyzing interest rates")
            results["interest_rates"] = self.analyze_interest_rates(interest_rate_files)

        return results

    def analyze_inflation_rates(self, inflation_rate_files: List[str]) -> Dict:
        """
        Analyzes inflation rates based on the provided files. It computes the average inflation rate, 
        identifies the trend, and calculates the annualized inflation rate.
        
        If the results are cached, they are loaded from the cache file.
        
        Args:
            inflation_rate_files (List[str]): List of file paths containing inflation rate data.
        
        Returns:
            Dict: Dictionary containing the analysis results, including average inflation, trend, 
                annualized inflation, and the last 12 months' inflation rates.
        """
        inflation_data = []
        file_name = os.path.basename(inflation_rate_files[0])
        cache_filename = f"cache/{file_name}_inflation_cache.json"
        
        # Check if cached data exists
        if os.path.exists(cache_filename):
            with open(cache_filename, "r") as cache_file:
                cached_result = json.load(cache_file)
                print(f"Using cached result for {file_name}")
                return cached_result  # Directly return cached result without filtering

        for file in inflation_rate_files:
            try:
                data = pd.read_csv(file)
                inflation_data.append(data)
                print(f"Inflation data loaded from {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        if not inflation_data:
            return {}

        df = pd.concat(inflation_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df = df.drop_duplicates(subset='Date', keep='last')

        df['Rate_of_Change'] = df['inflation_percentage'].pct_change() * 100
        average_inflation = round(df['inflation_percentage'].mean(), 2)
        
        trend = "Increasing" if df['inflation_percentage'].iloc[-12:].mean() > df['inflation_percentage'].iloc[-13] else "Decreasing"

        latest_inflation_rate = df['inflation_percentage'].iloc[-1] / 100
        annualized_inflation = round(float(((1 + latest_inflation_rate)**12 - 1) * 100), 2)

        df['inflation_percentage'] = df['inflation_percentage'].apply(lambda x: None if pd.isna(x) else x)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        last_12_months_inflation = df['inflation_percentage'].iloc[-12:].tolist()

        analysis_results = {
            "average_inflation": average_inflation,
            "trend": trend,
            "annualized_inflation": annualized_inflation,
            "last_12_months_inflation": last_12_months_inflation
        }

        # Save results to cache
        with open(cache_filename, "w") as cache_file:
            json.dump(analysis_results, cache_file)

        return analysis_results  # Return analysis results without filtering

    def analyze_fx_rates(self, fx_rate_files: List[str]) -> Dict:
        """
        Analyzes FX rates based on the provided files. It calculates average rates, rate of changes, trends,
        and the min/max for each currency. It returns only the relevant data for the specified currencies.
        
        If the results are cached, they are loaded from the cache file.
        
        Args:
            fx_rate_files (List[str]): List of file paths containing FX rate data.
        
        Returns:
            Dict: Dictionary containing the analysis results for the relevant currencies, including average rates, 
                rate of changes, trends, and min/max rates.
        """
        fx_data = []
        file_name = os.path.basename(fx_rate_files[0])
        cache_filename = f"cache/{file_name}_fx_cache.json"
        
        # Check if cached data exists
        if os.path.exists(cache_filename):
            with open(cache_filename, "r") as cache_file:
                cached_result = json.load(cache_file)
                print(f"Using cached result for {file_name}")
                return self.filter_and_convert_cached_data(cached_result)  # Only apply filtering here

        for file in fx_rate_files:
            try:
                data = pd.read_csv(file)
                fx_data.append(data)
                print(f"FX rate data loaded from {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        if not fx_data:
            return {}

        df = pd.concat(fx_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df = df.drop_duplicates(subset='Date', keep='last')

        rate_of_changes = {}
        average_rates = {}
        trends = {}
        min_max_rates = {}

        for currency in df.columns[1:]:
            df[currency] = pd.to_numeric(df[currency], errors='coerce')
            df[f'{currency}_rate_of_change'] = df[currency].pct_change() * 100
            rate_of_changes[currency] = df[f'{currency}_rate_of_change'].mean()

            average_rates[currency] = df[currency].mean()

            last_12_days = df[currency].iloc[-12:]
            trend = "Increasing" if last_12_days.mean() > df[currency].iloc[-13] else "Decreasing"
            trends[currency] = trend

            min_max_rates[currency] = {"min": df[currency].min(), "max": df[currency].max()}

        analysis_results = {
            "average_rates": average_rates,
            "rate_of_changes": rate_of_changes,
            "trends": trends,
            "min_max_rates": min_max_rates
        }

        # Save results to cache
        with open(cache_filename, "w") as cache_file:
            json.dump(analysis_results, cache_file)

        return self.filter_and_convert_cached_data(analysis_results)  # Apply filtering here as it’s specific to FX rates

    def filter_and_convert_cached_data(self, data: Dict) -> Dict:
        """Filter and convert cached data to only include relevant currencies and Python native types."""
        # Filter the data to include only the desired currencies
        filtered_data = {
            "average_rates": {currency: data["average_rates"][currency] for currency in self.currencies if currency in data["average_rates"]},
            "rate_of_changes": {currency: data["rate_of_changes"][currency] for currency in self.currencies if currency in data["rate_of_changes"]},
            "trends": {currency: data["trends"][currency] for currency in self.currencies if currency in data["trends"]},
            "min_max_rates": {currency: data["min_max_rates"][currency] for currency in self.currencies if currency in data["min_max_rates"]}
        }
        return self.convert_np_to_float(filtered_data)

    def convert_np_to_float(self, data):
        """Recursively convert np.float64 to native Python float."""
        if isinstance(data, dict):
            return {k: self.convert_np_to_float(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_np_to_float(item) for item in data]
        elif isinstance(data, (float, int)):
            return float(data)
        return data
    
    def analyze_interest_rates(self, interest_rate_files: List[str]) -> Dict:
        """
        Analyzes interest rates based on the provided files. Computes the average interest rate, rate of change, trend,
        and min/max values. If the results are cached, returns the cached data; otherwise, processes the data, computes
        the analysis, and caches the result.

        Args:
            interest_rate_files (List[str]): List of file paths containing interest rate data.

        Returns:
            Dict: Dictionary containing the analysis results, including the name of the interest rate series, average rate,
                rate of change, trend, min/max rates.
        """
        file_name = os.path.basename(interest_rate_files[0])
        cache_filename = f"cache/{file_name}_interest_rate_cache.json"

        print("calculating interest rates")
        # Check if cached data exists
        if os.path.exists(cache_filename):
            with open(cache_filename, "r") as cache_file:
                cached_result = json.load(cache_file)
                print(f"Using cached result for {file_name}")
                return cached_result

        # Load the data from the files
        interest_rate_data = []
        interest_rate_name = None

        for file in interest_rate_files:
            try:
                data = pd.read_csv(file)
                interest_rate_data.append(data)
                interest_rate_name = data.columns[1]  # Extract the name of the interest rate (e.g., SOFR, EUR_interest_rate)
                print(f"Interest rate data loaded from {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        if not interest_rate_data:
            return {}

        # Combine the data into one DataFrame
        df = pd.concat(interest_rate_data)
        date_columns = ['Date', 'observation_date']
        for col in date_columns:
            if col in df.columns:
                df['Date'] = pd.to_datetime(df[col])
                break
        df = df.sort_values(by='Date')  # Sort by date to ensure chronological order
        df = df.drop_duplicates(subset='Date', keep='last')  # Remove duplicates, keeping the latest data

        # Calculate Average Interest Rate
        average_interest_rate = df.iloc[:, 1].mean()

        # Calculate Rate of Change (Day-to-Day Interest Rate Change)
        df['Rate_of_Change'] = df.iloc[:, 1].pct_change() * 100
        rate_of_change = df['Rate_of_Change'].mean()

        # Calculate Trend (increasing or decreasing over the last 12 months)
        trend = "Increasing" if df.iloc[-12:, 1].mean() > df.iloc[-13, 1] else "Decreasing"

        # Calculate Min/Max Interest Rate
        min_interest_rate = df.iloc[:, 1].min()
        max_interest_rate = df.iloc[:, 1].max()

        # Prepare the analysis results
        analysis_results = {
            "interest_rate_name": interest_rate_name,
            "average_rate": average_interest_rate,
            "rate_of_change": rate_of_change,
            "trend": trend,
            "min_max_rates": {"min": min_interest_rate, "max": max_interest_rate}
        }

        # Save the result to cache
        with open(cache_filename, "w") as cache_file:
            json.dump(analysis_results, cache_file)

        return analysis_results

