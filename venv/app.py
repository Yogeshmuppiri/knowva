from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"PROVIDE ME AN ARTICLE ON MACHINE LEARNING"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)

