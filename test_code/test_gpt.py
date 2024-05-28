# 适用版本 openai <= 0.28.1

import openai
import llm.config as config

openai.api_key = config.GPT_API_KEY
openai.api_base = config.GPT_BASE_URL

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "你好"}])

# print the chat completion
print(chat_completion.choices[0].message.content)