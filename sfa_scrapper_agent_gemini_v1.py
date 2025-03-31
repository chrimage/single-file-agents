# /// script
# dependencies = [
#   "google-genai>=1.8.0",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0", # Pydantic is used by google-genai for schema validation
#   "firecrawl-py>=0.1.0",
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
    Example Usage:
        uv run sfa_scrapper_agent_gemini_v1.py -u "https://example.com" -p "Scrap and format each sentence as a separate line in a markdown list" -o "example.md"

        uv run sfa_scrapper_agent_gemini_v1.py \
            --url https://agenticengineer.com/principled-ai-coding \
            --prompt "What are the names and descriptions of each lesson?" \
            --output-file-path paic-lessons.md \
            -c 10
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types as genai_types # Corrected import
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Initialize Firecrawl
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    console.print(
        "[red]Error: FIRECRAWL_API_KEY not found in environment variables[/red]"
    )
    sys.exit(1)

firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# Initialize Gemini client
# Ensure GOOGLE_API_KEY environment variable is set
try:
    client = genai.Client()
except Exception as e:
    console.print(f"[red]Error initializing Gemini client: {e}[/red]")
    console.print(
        "[yellow]Please ensure the GOOGLE_API_KEY environment variable is set.[/yellow]"
    )
    sys.exit(1)

# --- Define Tools using Gemini's FunctionDeclaration ---

scrape_url_func = genai_types.FunctionDeclaration(
    name="scrape_url",
    description="Scrapes content from a URL and saves it to a file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explanation for why we're scraping this URL",
            ),
            "url": genai_types.Schema(
                type=genai_types.Type.STRING, description="The URL to scrape"
            ),
            "output_file_path": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Path to save the scraped content",
            ),
        },
        required=["reasoning", "url", "output_file_path"],
    ),
)

read_local_file_func = genai_types.FunctionDeclaration(
    name="read_local_file",
    description="Reads content from a local file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explanation for why we're reading this file",
            ),
            "file_path": genai_types.Schema(
                type=genai_types.Type.STRING, description="Path of the file to read"
            ),
        },
        required=["reasoning", "file_path"],
    ),
)

update_local_file_func = genai_types.FunctionDeclaration(
    name="update_local_file",
    description="Updates content in a local file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explanation for why we're updating this file",
            ),
            "file_path": genai_types.Schema(
                type=genai_types.Type.STRING, description="Path of the file to update"
            ),
            "content": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="New content to write to the file",
            ),
        },
        required=["reasoning", "file_path", "content"],
    ),
)

complete_task_func = genai_types.FunctionDeclaration(
    name="complete_task",
    description="Signals that the task is complete",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explanation of why the task is complete",
            )
        },
        required=["reasoning"],
    ),
)

# Create a Tool object containing all function declarations
gemini_tools = genai_types.Tool(
    function_declarations=[
        scrape_url_func,
        read_local_file_func,
        update_local_file_func,
        complete_task_func,
    ]
)

# --- Agent Prompt ---
# Note: The <tools> section here is descriptive for the LLM's understanding,
# but the actual tool definitions are passed via the API's `tools` parameter.
AGENT_PROMPT = """<purpose>
    You are a world-class web scraping and content filtering expert using Google Gemini.
    Your goal is to scrape web content and filter it according to the user's needs.
</purpose>

<instructions>
    <instruction>Follow this workflow: scrape_url -> read_local_file -> update_local_file. Repeat read/update as needed. Call complete_task when finished.</instruction>
    <instruction>When processing content, extract exactly what the user asked for - no more, no less.</instruction>
    <instruction>When saving processed content, use proper markdown formatting.</instruction>
    <instruction>Use the available function calls.</instruction>
</instructions>

<tools_description>
    - scrape_url: Scrapes a URL and saves raw content.
    - read_local_file: Reads content from a saved file for processing.
    - update_local_file: Updates the file with processed/filtered content.
    - complete_task: Signals the user's request is fully satisfied.
</tools_description>

<user_prompt>
    {{user_prompt}}
</user_prompt>

<url>
    {{url}}
</url>

<output_file_path>
    {{output_file_path}}
</output_file_path>
"""


# --- Helper Functions (Identical to OpenAI version) ---
def log_function_call(function_name: str, function_args: dict):
    """Log a function call in a rich panel."""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())
    console.print(
        Panel(
            f"{function_name}({args_str})",
            title="[blue]Function Call[/blue]",
            border_style="blue",
        )
    )


def log_function_result(function_name: str, result: str):
    """Log a function result in a rich panel."""
    console.print(
        Panel(
            str(result),
            title=f"[green]{function_name} Result[/green]",
            border_style="green",
        )
    )


def log_error(error_msg: str):
    """Log an error in a rich panel."""
    console.print(Panel(str(error_msg), title="[red]Error[/red]", border_style="red"))


def scrape_url(reasoning: str, url: str, output_file_path: str) -> str:
    """Scrapes content from a URL and saves it to a file."""
    log_function_call(
        "scrape_url",
        {"reasoning": reasoning, "url": url, "output_file_path": output_file_path},
    )
    try:
        # Corrected Firecrawl parameters based on API documentation/common usage
        response = firecrawl_app.scrape_url(
            url=url,
            params={
                "onlyMainContent": True,
                "formats": ["markdown"], # Using 'formats' as seen in OpenAI example and MCP docs
            },
        )
        # Check response structure (may vary based on library version/API)
        content = None
        if isinstance(response, dict) and response.get("markdown"):
            content = response["markdown"]
        elif isinstance(response, list) and response and isinstance(response[0], dict) and response[0].get("markdown"):
            # Handle potential list response like older versions
            content = response[0]["markdown"]

        if content:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            result_msg = f"Successfully scraped {len(content)} characters from {url} to {output_file_path}"
            log_function_result("scrape_url", result_msg)
            # Return the content for potential immediate use if needed, though the agent flow reads it back
            return content
        else:
            error_msg = f"Error scraping URL {url}: No markdown content found or unexpected response format: {response}"
            log_error(error_msg)
            return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error scraping URL {url}: {str(e)}"
        log_error(error_msg)
        return f"Error: {error_msg}"


def read_local_file(reasoning: str, file_path: str) -> str:
    """Reads content from a local file."""
    log_function_call(
        "read_local_file", {"reasoning": reasoning, "file_path": file_path}
    )
    try:
        console.log(
            f"[blue]Reading File[/blue] - File: {file_path} - Reasoning: {reasoning}"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        log_function_result("read_local_file", f"Read {len(content)} characters")
        return content
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        log_error(error_msg)
        return f"Error: {error_msg}"


def update_local_file(reasoning: str, file_path: str, content: str) -> str:
    """Updates content in a local file."""
    log_function_call(
        "update_local_file",
        {
            "reasoning": reasoning,
            "file_path": file_path,
            "content": f"{len(content)} characters",  # Don't log full content
        },
    )
    try:
        console.log(
            f"[blue]Updating File[/blue] - File: {file_path} - Reasoning: {reasoning}"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        result_msg = f"Successfully wrote {len(content)} characters to {file_path}"
        log_function_result("update_local_file", result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"Error updating file {file_path}: {str(e)}"
        log_error(error_msg)
        return f"Error: {error_msg}"


def complete_task(reasoning: str) -> str:
    """Signals that the task is complete."""
    log_function_call("complete_task", {"reasoning": reasoning})
    console.log(f"[green]Task Complete[/green] - Reasoning: {reasoning}")
    result = "Task completed successfully"
    log_function_result("complete_task", result)
    return result


# Map function names to actual functions
available_functions = {
    "scrape_url": scrape_url,
    "read_local_file": read_local_file,
    "update_local_file": update_local_file,
    "complete_task": complete_task,
}

# --- Main Execution Logic ---
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Web scraper agent (Gemini) that filters content based on user query"
    )
    parser.add_argument("--url", "-u", required=True, help="The URL to scrape")
    parser.add_argument(
        "--output-file-path",
        "-o",
        default="scraped_content.md",
        help="Path to save the scraped content",
    )
    parser.add_argument(
        "--prompt", "-p", required=True, help="The prompt to filter the content with"
    )
    parser.add_argument(
        "--compute-limit",
        "-c",
        type=int,
        default=10,
        help="Maximum number of agent iterations",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.0-flash", # Updated default model
        help="Gemini model to use (e.g., gemini-2.0-flash, gemini-1.5-pro)",
    )

    args = parser.parse_args()

    # Format the initial prompt
    formatted_prompt = (
        AGENT_PROMPT.replace("{{user_prompt}}", args.prompt)
        .replace("{{url}}", args.url)
        .replace("{{output_file_path}}", args.output_file_path)
    )

    # Initialize conversation history (list of Content objects)
    messages: List[genai_types.Content] = [genai_types.Content(role="user", parts=[genai_types.Part(text=formatted_prompt)])]

    # Track number of iterations
    iterations = 0
    max_iterations = args.compute_limit
    break_loop = False

    while iterations < max_iterations:
        if break_loop:
            break

        iterations += 1
        console.rule(f"[yellow]Agent Loop {iterations}/{max_iterations}[/yellow]")

        try:
            # Send message to Gemini
            response = client.models.generate_content(
                model=args.model,
                contents=messages,
                config=genai_types.GenerateContentConfig(tools=[gemini_tools]) # Correct parameter name is 'config'
                # config=genai_types.GenerateContentConfig(temperature=0.0, tools=[gemini_tools]) # Example if also setting temp
            )

            # Extract the response candidate
            candidate = response.candidates[0]
            # Handle cases where the response might be blocked or finish unexpectedly
            if not candidate.content or not candidate.content.parts:
                console.print("[yellow]Warning: Received empty or incomplete response candidate.[/yellow]")
                # Potentially add a placeholder or error message to history
                messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text="[Model response was empty or blocked]")]))
                continue # Skip processing this turn

            response_part = candidate.content.parts[0] # Assuming single part response for now

            # Print assistant's text response if any
            if response_part.text:
                console.print(Panel(response_part.text, title="[magenta]Assistant[/magenta]"))
                # Add assistant's text response to history
                messages.append(genai_types.Content(role="model", parts=[response_part]))
            elif response_part.function_call:
                # If only a function call, add a placeholder model message before the function call part
                messages.append(genai_types.Content(role="model", parts=[response_part]))
            else:
                console.print("[yellow]Assistant response did not contain text or function call.[/yellow]")
                # Add the raw response part to history anyway
                messages.append(genai_types.Content(role="model", parts=[response_part]))


            # Handle function calls
            if response_part.function_call:
                fc = response_part.function_call
                function_name = fc.name
                function_args = dict(fc.args) # Convert FunctionCall.args (Struct) to dict

                console.print(
                    Panel(
                        f"Processing tool call: {function_name}({function_args})",
                        title="[yellow]Tool Call[/yellow]",
                        border_style="yellow",
                    )
                )

                # Execute the function
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    try:
                        function_response_content = function_to_call(**function_args)

                        # Check if the task is complete
                        if function_name == "complete_task":
                            break_loop = True

                        # Add function response to history
                        messages.append(
                            genai_types.Content(
                                role="function", # Note: Gemini uses 'function' role for responses
                                parts=[
                                    genai_types.Part.from_function_response(
                                        name=function_name,
                                        response={"result": function_response_content}, # Wrap result in a dict
                                    )
                                ],
                            )
                        )

                    except Exception as e:
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        log_error(error_msg)
                        # Add error response to history
                        messages.append(
                             genai_types.Content(
                                role="function",
                                parts=[
                                    genai_types.Part.from_function_response(
                                        name=function_name,
                                        response={"error": error_msg},
                                    )
                                ],
                            )
                        )
                        # Potentially break or let the model try to recover? For now, continue.
                else:
                    log_error(f"Unknown function call requested: {function_name}")
                    # Add error response to history
                    messages.append(
                         genai_types.Content(
                            role="function",
                            parts=[
                                genai_types.Part.from_function_response(
                                    name=function_name,
                                    response={"error": f"Unknown function: {function_name}"},
                                )
                            ],
                        )
                    )
            elif not response_part.text:
                # If there was no text and no function call, something might be wrong
                console.print("[yellow]Warning: Assistant response had neither text nor function call.[/yellow]")
                # Let the loop continue, maybe the model will recover

        except Exception as e:
            log_error(f"An error occurred during the agent loop: {str(e)}")
            console.print("[yellow]Current Conversation History:[/yellow]")
            # Safely print history parts
            for msg in messages:
                try:
                    console.print(msg)
                except Exception as print_e:
                    console.print(f"[red]Error printing message: {print_e}[/red]")
            break # Stop the loop on general errors

    if iterations >= max_iterations and not break_loop:
        log_error("Reached maximum number of iterations without completing the task.")
        # Consider raising an exception or just exiting
        # raise Exception("Reached maximum number of iterations")

    console.print("[bold green]Agent finished.[/bold green]")


if __name__ == "__main__":
    main()
