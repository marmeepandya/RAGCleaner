# import PyDI
# print(dir(PyDI))

from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);