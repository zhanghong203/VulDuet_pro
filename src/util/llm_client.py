from openai import OpenAI


class LLMClient:

    def __init__(
            self,
            api_key,
            base_url,
            model,
            temperature=0.0
    ):

        self.model = model
        self.temperature = temperature

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str) -> str:
        """
        输入Prompt
        返回LLM生成结果
        """
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": 1.0,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = self.client.chat.completions.create(
            **params
        )

        return response.choices[0].message.content
