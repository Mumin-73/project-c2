from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
ASSISTANT_ID = os.environ['OPENAI_ASSISTANT_ID']

client = OpenAI(api_key=API_KEY)

def get_openai_response(transcribed_text):
    # Create a thread
    thread = client.beta.threads.create()

    # Create a message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=transcribed_text
    )

    # Execute assistant and get result
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )

    # Get message list
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Iterate through messages to get the last one
    response_text = ""
    for msg in messages.data:
        response_content = msg.content
        if isinstance(response_content, str):
            response_text += response_content
        elif isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, 'text'):
                    response_text += str(block.text)
                elif hasattr(block, 'content'):  # Handle different response formats
                    response_text += block.content

    if response_text is None:
        raise ValueError("No messages found in thread")

    return response_text