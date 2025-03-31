# Guide to Using the `google-genai` Python Library (v1.8.0) for LLMs

This guide provides essential information for generating Python code using the `google-genai` library (version 1.8.0). It focuses on core concepts, common patterns, and key components based on the official documentation scraped from [https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/).

**Version:** This guide specifically targets `google-genai` **v1.8.0**. Features and APIs might differ in other versions.

---

## **IMPORTANT NOTE: `google-genai` vs. `google-generativeai`**

*   **Use `google-genai`:** This is the **current, recommended library** for interacting with Google's generative AI models (Gemini, Imagen, etc.) via Python, supporting the Gemini API.
*   **Avoid `google-generativeai`:** This is an **older, legacy library** specifically for the Gemini API. It lacks support for newer capabilities found in `google-genai`.
*   **Why Referencing Documentation is CRUCIAL:** Your training data likely contains examples from *both* libraries, and potentially incorrect usage patterns even for `google-genai`. **Always prioritize patterns and examples shown in the official `google-genai` documentation** ([https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/)) over patterns recalled from training data to ensure correctness and leverage the latest features. The `google-genai` library uses different class/method names and structures compared to the older library.

---

## Core Concepts

### 1. Client Initialization

The primary entry point is `genai.Client`. Initialization depends on the target API:

*   **Gemini API (Developer API):** Requires an API key.
    ```python
    from google import genai

    client = genai.Client(api_key="YOUR_API_KEY")
    ```
*   **Using Environment Variable:** Alternatively, set the `GOOGLE_API_KEY` environment variable. The client will automatically pick it up.
    ```bash
    # In your terminal or environment configuration
    export GOOGLE_API_KEY='YOUR_API_KEY'
    ```
    ```python
    # Python code
    from google import genai
    import os
    # Ensure the variable is set before initializing
    # Consider using a library like python-dotenv to load from a .env file
    # Example:
    # from dotenv import load_dotenv
    # load_dotenv()
    client = genai.Client()
    ```
    *(**Best Practice:** Use environment variables, potentially loaded from a `.env` file using libraries like `python-dotenv`, to avoid hardcoding keys in your source code.)*

### 2. Asynchronous Client

For non-blocking operations, access the async client via the `.aio` property:
```python
async_client = client.aio
# Now use async methods, e.g., await async_client.models.generate_content(...)
```

### 3. Service Modules

Functionality is organized into modules accessible via the client instance (e.g., `client.models`, `client.chats`). Key modules include:

*   `client.models`: Core interactions like text/image/video generation, embedding, token counting.
*   `client.chats`: Managing conversational interactions.
*   `client.files`: Uploading, downloading, and managing files for use with models (primarily Gemini API).
*   `client.tunings`: Managing model fine-tuning jobs.
*   `client.live`: (Experimental) Real-time interaction sessions.
*   `client.operations`: Managing long-running operations (like tuning or batch jobs).

### 4. Input/Output Structure (`Content` and `Part`)

Input (`contents`) and output (`candidates[].content`) use a structured format:

*   `Content`: Represents a single message turn (e.g., a user prompt or a model response). Contains a `role` (`'user'` or `'model'`) and a list of `parts`.
*   `Part`: Represents a piece of data within a `Content` object. Can be text, inline data (bytes + mime_type), file data (URI + mime_type), function calls/responses, etc.
    *   Use helper methods like `Part.from_text()`, `Part.from_uri()`, `Part.from_data()`.
    *   Multimodal input involves multiple `Part` objects within a single `Content`.

```python
from google.genai import types

# Simple text input
user_prompt = "Explain quantum physics simply."

# Multimodal input
user_prompt_multi = types.Content(parts=[
    types.Part.from_text("Describe this image:"),
    types.Part.from_uri("gs://cloud-samples-data/generative-ai/image/scones.jpg", "image/jpeg")
])

# Or using helper constructors
user_prompt_multi_alt = types.UserContent([
    "Describe this image:",
    types.Part.from_uri("gs://cloud-samples-data/generative-ai/image/scones.jpg", "image/jpeg")
])
```

### 5. Configuration Objects (`types` module)

Behavior is controlled via configuration objects passed to methods (often in a `config` dictionary parameter). Key examples:

*   `types.GenerationConfig`: Controls generation parameters (temperature, top_k, top_p, max_output_tokens, stop_sequences, safety_settings, tools, tool_config, response_mime_type, response_schema, etc.).
*   `types.SafetySetting`: Defines thresholds for blocking harmful content (`category`, `threshold`).
*   `types.Tool`: Defines tools the model can use (Function Declarations, Retrieval, Google Search).
*   `types.FunctionDeclaration`: Defines a function the model can call (name, description, parameters schema, response schema).
*   `types.EmbedContentConfig`: Configures embedding requests (task_type, title, output_dimensionality).
*   `types.HttpOptions`: Overrides HTTP client settings (timeout, headers, api_version).

---

## Common Tasks (Concise Examples)

*(Assume `client` is an initialized `genai.Client`)*

**1. Generate Text:**
```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-1.5-flash", # Or other appropriate model
    contents="Write a short poem about the moon.",
    config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=100
    )
)
print(response.text)
```

**2. Stream Text:**
```python
from google.genai import types

stream = client.models.generate_content_stream(
    model="gemini-1.5-flash",
    contents="Tell me a long story about a brave knight."
)
for chunk in stream:
    print(chunk.text, end="")
print()
```

**3. Start a Chat:**
```python
chat = client.chats.create(model="gemini-1.5-flash")
response = chat.send_message("Hello! What can you do?")
print(response.text)

response2 = chat.send_message("What was the first thing I said?")
print(response2.text)
# `chat.history` contains the conversation turns
```

**4. Embed Content:**
```python
from google.genai import types

response = client.models.embed_content(
    model="text-embedding-004", # Or other embedding model
    contents=["What is the meaning of life?", "How does photosynthesis work?"],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
# response.embeddings is a list of ContentEmbedding objects
print(len(response.embeddings))
print(len(response.embeddings[0].values))
```

**5. Count Tokens:**
```python
response = client.models.count_tokens(
    model="gemini-1.5-flash",
    contents="How many tokens are in this sentence?"
)
print(f"Total tokens: {response.total_tokens}")
```

**6. Upload a File (Gemini API):**
```python
# Note: File API is primarily for the Gemini API (ai.google.dev).
try:
    # Requires API Key client
    file_response = client.files.upload(
        file="path/to/your/image.jpg",
        config=types.UploadFileConfig(display_name="My Uploaded Image")
    )
    print(f"Uploaded file: {file_response.name}, URI: {file_response.uri}")
    # Use file_response.uri or file_response itself in subsequent prompts
except Exception as e:
     print(f"File upload likely requires Gemini API client: {e}")

# Example using the uploaded file URI in a prompt
# prompt_with_file = types.Content(parts=[
#     types.Part.from_text("Describe this uploaded image:"),
#     types.Part.from_uri(file_response.uri, mime_type=file_response.mime_type)
# ])
# gen_response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt_with_file)
```

**7. Using Function Calling:**
```python
from google.genai import types

# Define the function schema
find_theaters = types.FunctionDeclaration(
    name="find_theaters",
    description="Find theaters showing movies based on location and optionally movie title.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location": types.Schema(type=types.Type.STRING, description="The city and state, e.g. San Francisco, CA"),
            "movie": types.Schema(type=types.Type.STRING, description="Optional. The name of the movie")
        },
        required=["location"]
    )
)

# Create a Tool object
tools = types.Tool(function_declarations=[find_theaters])

# Make the generate_content call with the tool
response = client.models.generate_content(
    model="gemini-1.5-flash", # Model must support function calling
    contents="Find me movie theaters in Mountain View, CA.",
    config=types.GenerateContentConfig(tools=[tools])
)

# Check for function call in response
if response.function_calls:
    fc = response.function_calls[0]
    print(f"Function Call: {fc.name}")
    print(f"Args: {fc.args}")
    # Here you would execute the function based on fc.name and fc.args
    # and send back a FunctionResponse Part in the next turn.
```

---

## Key Types (`google.genai.types`)

The `types` module is crucial. It contains Pydantic models and Enums for configuring requests and parsing responses. Explore this module in the documentation for detailed structures like:

*   `Content`, `Part`, `Blob`, `FileData`
*   `GenerateContentConfig`, `SafetySetting`, `HarmCategory`, `HarmBlockThreshold`
*   `Tool`, `FunctionDeclaration`, `FunctionCall`, `FunctionResponse`, `Schema`, `Type`
*   `BatchJob`, `CachedContent`, `File`, `Model`, `TuningJob`
*   `GenerateContentResponse`, `EmbedContentResponse`, `CountTokensResponse`, etc.

---

**Reminder:** Always consult the official `google-genai` documentation for the most up-to-date information, detailed parameter descriptions, and advanced usage patterns.
