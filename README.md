# GPTChat Pyhton Class

## Overview
The `GPTChat` class is a Python class designed to interface with OpenAI's GPT-4 language model to generate conversational responses. It utilizes OpenAI's API for generating responses and manages a conversation's state through a chat history.

## Features
- Utilizes OpenAI's GPT-4 model for generating responses.
- Manages chat history to provide context for generating responses.
- Allows customization of the model's temperature and top_p sampling parameters.
- Limits the token count of the conversation history to fit within GPT-4's maximum token limits.

## Installation
To use the `GPTChat` class, you must have the `openai` and `transformers` Python packages installed.

`pip install openai transformers`


## Usage
To begin a chat session, create an instance of the `GPTChat` class with your OpenAI API key. Then, use the `send_message` method to send messages to the GPT-4 model and receive generated responses.

```python
from gpt_chat import GPTChat

api_key = "your-api-key"
chat_session = GPTChat(api_key)

response = chat_session.send_message(
    message="Hello, how are you?",
    systemContent="You are a helpful assistant",
    temperature=0.7,
    top_p=0.9
)
print(response)
