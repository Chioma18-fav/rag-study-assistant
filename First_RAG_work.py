import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(api_key= API_KEY, base_url="https://openrouter.ai/api/v1")

def load_text_file():
    with open("sample_notes_plain.txt", "r", encoding="utf-8") as notes:
        content = notes.read()
        return content

contents = load_text_file()
question = input("Enter ur question: ")

prompt = f""" 
Use the following notes to answer the question, answer concisely, in plain text.
   {contents}
   Question: {question}
   """

message = [
    {'role': 'user', 'content': prompt}]

response = client.chat.completions.create(
    messages=message, 
    model="openai/gpt-oss-20b:free", 
)

print(response.choices[0].message.content)