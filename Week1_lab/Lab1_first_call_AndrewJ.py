from dotenv import load_dotenv 
load_dotenv()
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
model="gpt-5.2", input="in 10 words or less, who is the strongest avenger?"
)
print(response.output_text)