import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')
OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')

def call_llm(prompt: str, max_tokens: int = 1000, expect_json: bool = False) -> str:
    if PROVIDER == 'ollama':
        return _call_ollama(prompt, max_tokens, expect_json)
    elif PROVIDER == 'claude':
        return _call_claude(prompt, max_tokens)
    elif PROVIDER == 'groq':
        return _call_groq(prompt, max_tokens, expect_json)
    else:
        raise ValueError(f'Unknown LLM provider: {PROVIDER}')

def _call_ollama(prompt: str, max_tokens: int, expect_json: bool) -> str:
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': max_tokens,
            'temperature': 0.7,
        }
    }
    if expect_json:
        payload['format'] = 'json'
    
    try:
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()['response'].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            'Cannot connect to Ollama. '
            'Make sure Ollama is running.'
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            'Ollama took too long. '
            'Try a smaller model in .env'
        )

def _call_claude(prompt: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    message = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return message.content[0].text.strip()

def _call_groq(prompt: str, max_tokens: int, expect_json: bool) -> str:
    from groq import Groq
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError('GROQ_API_KEY is not set in .env')

    client = Groq(api_key=api_key)
    request_args = {
        'model': GROQ_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.7,
    }
    if expect_json:
        request_args['response_format'] = {'type': 'json_object'}

    completion = client.chat.completions.create(**request_args)
    return completion.choices[0].message.content.strip()
