import json
from openai import OpenAI
from tqdm import tqdm


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
            api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
        )

        completion = client.chat.completions.create(
          model="Qwen2.5-7B-Instruct",
          messages=[
            {"role": "user", "content": query}
          ],
          temperature=self.params['temperature'],
          max_tokens=self.params['max_new_tokens'],
          top_p=self.params['top_p']
        )

        return completion.choices[0].message.content


if __name__ == "__main__":
    qwen = LLM_Model("Qwen2.5-7B-Instruct")
    data_path = "/mnt/workspace/dataset/HotpotQA/dev.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefix = "Given the following question, please think carefully and answer the questions raised simply and directly. Question: "
    results = []
    for item in tqdm(data):
        query = prefix + item['question']
        try:
            response = qwen.request(query)
        except Exception as e:
            response = "No Answer"
            print("id: " + item['_id'], e)
        results.append({
            "question": item['question'],
            'answer': item['answer'],
            'prediction': response
        })
    output_path = '/mnt/workspace/results/HotpotQA/qwen2.5-7B-Instruct-only.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f)