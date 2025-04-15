from abc import ABC, abstractmethod
from typing import List, Dict
from LLM_API import LLM_API
#Author: Erik Schnell

class BaseAnalyzer(ABC):
    def __init__(self, file_names: List[str]):
        """
        Abstract base class for all analyzers.
        Shared access to the LLM API is provided via self.llm.
        """
        self.file_names = file_names
        self.llm = LLM_API()

    @abstractmethod
    def analyze(self) -> Dict:
        """
        Each analyzer must implement this method.
        Returns a structured JSON/dictionary output.
        """
        pass


