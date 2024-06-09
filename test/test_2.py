from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')
client = OpenAI(api_key=API_KEY)

def generate_response(prompt, thread_id=None):
    if thread_id:
        thread = client.beta.threads.get(thread_id)
    else:
        thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    response_text = ""
    for msg in messages.data:
        if msg.role == "user":
            continue
        elif msg.role == "assistant":
            for content_block in msg.content:
                if content_block.type == "text":
                    text_obj = content_block.text
                    response_text += text_obj.value

    print(response_text)

if __name__ == "__main__":
    prompt = input()
    response = generate_response(prompt)