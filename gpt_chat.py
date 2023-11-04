import os
import openai
from transformers import GPT2TokenizerFast

# Set the tokenizers parallelism environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GPTChat:
    """
    A class for generating responses to messages using OpenAI's GPT-4 language model.
    
    Attributes:
        api_key (str): The OpenAI API key to use for authentication.
        completion_model (str): The name of the GPT model to use for generating responses.
        chat_history (list): A list of tuples representing the chat history, where each tuple contains the prompt and response.
        
    Methods:
        send_message(message, systemContent, temperature, top_p, max_history): Sends a message to the GPT-4 model and returns the generated response.
    """

    def __init__(self, api_key):
        """
        Initializes a new instance of the GPTChat class.

        Args:
            api_key (str): The OpenAI API key to use for authentication.
        """
        openai.api_key = api_key
        self.completion_model = "gpt-4"
        self.chat_history = []
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 8192  # Set the tokenizer maximum length to match GPT-4's limit

    def send_message(self, message, systemContent, temperature, top_p, max_history=5):
        """
        Sends a message to the GPT-4 chat model and returns the generated response.

        Args:
            message (str): The message to send to the GPT-4 chat model.
            systemContent (str): System-level content that provides context for the model.
            temperature (float): Sampling temperature to use for generating responses.
            top_p (float): Nucleus sampling top p value for generating responses.
            max_history (int): The maximum number of message-response pairs to include in the context.

        Returns:
            str: The generated response from the GPT-4 chat model.
        """

        # Prepare the messages in the format expected by the ChatCompletion API
        messages_to_send = [
            {"role": "system", "content": systemContent},
            {"role": "user", "content": message}
        ]

        # Add the history to the messages to send, maintaining the order
        history = self.chat_history[-max_history:]
        for prompt, response in history:
            messages_to_send.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ])

        # Calculate the number of tokens used by the messages
        message_token_lengths = [len(self.tokenizer.encode(m['content'])) for m in messages_to_send]
        total_message_tokens = sum(message_token_lengths)

        # Calculate the remaining tokens for the model's response
        remaining_tokens = 8192 - total_message_tokens - 100

        # Ensure that the response does not exceed the remaining token count
        max_response_tokens = remaining_tokens

        # Send the message to the API
        try:
            response = openai.ChatCompletion.create(
                model=self.completion_model,
                temperature=temperature,
                top_p=top_p,
                messages=messages_to_send,
                max_tokens=max_response_tokens
            )
        except openai.error.InvalidRequestError as e:
            print(f"An error occurred: {e}")
            return None

        # Extract the response content
        message_response = response.choices[0].message['content'].strip()

        # Update the chat history
        self.chat_history.append((message, message_response))

        # Truncate the chat history if necessary
        if len(self.chat_history) > max_history * 2:
            self.chat_history = self.chat_history[-max_history * 2:]

        return message_response
