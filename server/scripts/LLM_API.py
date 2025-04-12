from openai import OpenAI
import base64

class LLM_API:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLM_API, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.client = OpenAI(
            api_key="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"
        )

    def ask(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={ "type":"json_object" },
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def ask_with_image(self, prompt: str, filepath: str) -> str:
        try:
            base64_image = self.encode_image(filepath)
            response = self.client.responses.create(
                model="gpt-4o",  # Assuming this is the correct model for image input
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    }
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            return f"Error: {e}"
        
    
    def ask_with_pdf(self, prompt: str, pdf_path: str) -> str:
        try:
            # Upload PDF and get the file ID
            file = self.client.files.create(
                file=open(pdf_path, "rb"),
                purpose="user_data"
            )
            

            # Query with file ID and text prompt
            response = self.client.responses.create(
                model="gpt-4o",  # Adjust the model name as per your use case
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file.id},
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            return f"Error: {e}"