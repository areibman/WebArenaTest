"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
import openai.error
from tqdm.asyncio import tqdm_asyncio
from agentops import Client
AGENTOPS_KEY='31453e35-43d8-43e6-98f6-5753893b2f19'
ao_client = Client(api_key=AGENTOPS_KEY, tags=['WebArena', 'Alex Local'])
print("AgentOps client created.")


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple[Any] = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""
    print("Starting retry_with_exponential_backoff function.")

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        print("Initialized variables.")

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                print("Trying to execute function.")
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                print(f"Caught error: {e}. Incremented retries to {num_retries}.")

                # Check if max retries has been reached
                if num_retries > max_retries:
                    print(f"Maximum number of retries ({max_retries}) exceeded.")
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Incremented delay to {delay}.")

                # Sleep for the delay
                print("Sleeping for the delay.")
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print(f"Caught unexpected error: {e}. Raising exception.")
                raise e

    return wrapper
    print("Returning wrapper.")


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    print("Starting _throttled_openai_completion_acreate function.")
    async with limiter:
        for _ in range(3):
            try:
                print("Trying to create OpenAI completion.")
                return await openai.Completion.acreate(  # type: ignore
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                print("Caught RateLimitError. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                print(f"Caught APIError: {e}. Breaking loop.")
                break
        print("Returning default response.")
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    print("Starting agenerate_from_openai_completion function.")
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables.")
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("Set OpenAI API key.")

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    print("Created AsyncLimiter.")
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    print("Created list of async responses.")
    responses = await tqdm_asyncio.gather(*async_responses)
    print("Gathered responses.")
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    print("Starting generate_from_openai_completion function.")
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables.")
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("Set OpenAI API key.")
    response = openai.Completion.create(  # type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    print("Created OpenAI completion.")
    answer: str = response["choices"][0]["text"]
    print("Extracted answer from response.")
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    print("Starting _throttled_openai_chat_completion_acreate function.")
    async with limiter:
        for _ in range(3):
            try:
                print("Trying to create OpenAI chat completion.")
                return await openai.ChatCompletion.acreate(  # type: ignore
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                print("Caught RateLimitError. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                print("Caught TimeoutError. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                print(f"Caught APIError: {e}. Breaking loop.")
                break
        print("Returning default response.")
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    print("Starting agenerate_from_openai_chat_completion function.")
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables.")
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("Set OpenAI API key.")

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    print("Created AsyncLimiter.")
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    print("Created list of async responses.")
    responses = await tqdm_asyncio.gather(*async_responses)
    print("Gathered responses.")
    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    print("Starting generate_from_openai_chat_completion function.")
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables.")
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("Set OpenAI API key.")

    response = openai.ChatCompletion.create(  # type: ignore
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    print("Created OpenAI chat completion.")
    answer: str = response["choices"][0]["message"]["content"]
    print("Extracted answer from response.")
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    print("Starting fake_generate_from_openai_chat_completion function.")
    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found in environment variables.")
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("Set OpenAI API key.")
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    print("Set answer.")
    return answer
