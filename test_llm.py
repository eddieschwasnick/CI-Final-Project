import os
import yaml
import anthropic
from dotenv import load_dotenv

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load API Key
load_dotenv(config['api_key_path'])
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Test Prompt
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, is this working?"}]
)

print(response.content[0].text)