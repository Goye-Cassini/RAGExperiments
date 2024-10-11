from openai import OpenAI


class LLM_Model():
    def __init__(self, model_name, temperature=1.0, max_new_tokens=2000, top_p=1.0):
        self.modelName = model_name
        self.params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p
        }

    def request(self, query: str) -> str:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123", # 随便填写，只是为了通过接口参数校验
        )
        completion = client.chat.completions.create(
          model=self.modelName,
          messages=[
            {"role": "user", "content": query}
          ],
          # temperature=self.params['temperature'],
          # max_tokens=self.params['max_new_tokens'],
          # top_p=self.params['top_p']
        )
        return completion.choices[0].message.conten