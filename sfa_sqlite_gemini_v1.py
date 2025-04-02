#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "google-genai>=1.8.0",
#   "rich>=13.7.0",
#   "pydantic>=2.0.0", # Required by google-genai
#   "python-dotenv>=1.0.0",
# ]
# ///

"""
Single-file agent for interacting with SQLite databases using Google Gemini.

Usage:
    uv run sfa_sqlite_gemini_v1.py --db data/analytics.sqlite --prompt "How many tables are there?"
    uv run sfa_sqlite_gemini_v1.py --db data/analytics.sqlite --prompt "Show me the schema for the 'users' table."
    uv run sfa_sqlite_gemini_v1.py --db data/analytics.sqlite --prompt "What are the first 3 users sorted by signup_date?"
"""

import os
import sys
import argparse
import json
import traceback
import sqlite3
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types as genai_types
# Note: FunctionCallingMode is not explicitly needed for GenerateContentConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Global variable for database path
DB_PATH: Optional[str] = None

# --- Define Tools using Gemini's FunctionDeclaration ---

list_tables_func = genai_types.FunctionDeclaration(
    name="list_tables",
    description="Returns list of available tables in database",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why we need to list tables relative to user request",
            ),
        },
        required=["reasoning"],
    ),
)

describe_table_func = genai_types.FunctionDeclaration(
    name="describe_table",
    description="Returns schema info for specified table",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why we need to describe this table",
            ),
            "table_name": genai_types.Schema(
                type=genai_types.Type.STRING, description="Name of table to describe"
            ),
        },
        required=["reasoning", "table_name"],
    ),
)

sample_table_func = genai_types.FunctionDeclaration(
    name="sample_table",
    description="Returns sample rows from specified table, always specify row_sample_size",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why we need to sample this table",
            ),
            "table_name": genai_types.Schema(
                type=genai_types.Type.STRING, description="Name of table to sample"
            ),
            "row_sample_size": genai_types.Schema(
                type=genai_types.Type.INTEGER,
                description="Number of rows to sample aim for 3-5 rows",
            ),
        },
        required=["reasoning", "table_name", "row_sample_size"],
    ),
)

run_test_sql_query_func = genai_types.FunctionDeclaration(
    name="run_test_sql_query",
    description="Tests a SQL query and returns results (only visible to agent)",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why we're testing this specific query",
            ),
            "sql_query": genai_types.Schema(
                type=genai_types.Type.STRING, description="The SQL query to test"
            ),
        },
        required=["reasoning", "sql_query"],
    ),
)

run_final_sql_query_func = genai_types.FunctionDeclaration(
    name="run_final_sql_query",
    description="Runs the final validated SQL query and shows results to user",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Final explanation of how query satisfies user request",
            ),
            "sql_query": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="The validated SQL query to run",
            ),
        },
        required=["reasoning", "sql_query"],
    ),
)

# Create a Tool object containing all function declarations
gemini_tools = genai_types.Tool(
    function_declarations=[
        list_tables_func,
        describe_table_func,
        sample_table_func,
        run_test_sql_query_func,
        run_final_sql_query_func,
    ]
)

# --- Agent Prompt ---
AGENT_PROMPT = """<purpose>
    You are a world-class expert at crafting precise SQLite SQL queries using Google Gemini.
    Your goal is to generate accurate queries that exactly match the user's data needs.
</purpose>

<instructions>
    <instruction>Use the provided tools to explore the database and construct the perfect query.</instruction>
    <instruction>Start by listing tables to understand what's available.</instruction>
    <instruction>Describe tables to understand their schema and columns.</instruction>
    <instruction>Sample tables to see actual data patterns.</instruction>
    <instruction>Test queries using `run_test_sql_query` before finalizing them.</instruction>
    <instruction>Only call `run_final_sql_query` when you're confident the query is perfect and satisfies the user request.</instruction>
    <instruction>Be thorough but efficient with tool usage.</instruction>
    <instruction>If your `run_test_sql_query` tool call returns an error or the results don't satisfy the user request, analyze the issue, fix the query, and test again. Do not proceed to `run_final_sql_query` if the test query failed or was incorrect.</instruction>
    <instruction>Think step by step about what information you need.</instruction>
    <instruction>Provide reasoning with every tool call.</instruction>
    <instruction>Tool results are returned as simple strings.</instruction>
</instructions>

<tools_description>
    - list_tables: Returns a comma-separated string list of available tables in database.
    - describe_table: Returns schema info for specified table as a multi-line string.
    - sample_table: Returns sample rows from specified table as a multi-line string.
    - run_test_sql_query: Tests a SQL query and returns results as a multi-line string (only visible to agent). Returns error message on failure.
    - run_final_sql_query: Runs the final validated SQL query and shows results to user as a multi-line string. Returns error message on failure. This is the final step.
</tools_description>

<user_request>
    {{user_request}}
</user_request>
"""

# --- Helper Functions for Logging ---
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

def log_function_result(function_name: str, result: str): # Now expects string
    """Log a function result (string) in a rich panel."""
    # Truncate long results for cleaner logging
    max_len = 500
    display_result = result
    if len(result) > max_len:
        display_result = result[:max_len] + f"... (truncated {len(result) - max_len} chars)"

    console.print(
        Panel(
            display_result,
            title=f"[green]{function_name} Result[/green]",
            border_style="green",
        )
    )

def log_error(error_msg: str):
    """Log an error in a rich panel."""
    console.print(Panel(str(error_msg), title="[red]Error[/red]", border_style="red"))

# --- Tool Implementation Functions ---

# Modified to return simple strings or error strings
def tool_list_tables(reasoning: str) -> str:
    """
    Returns a comma-separated string list of tables in the database.
    """
    log_function_call("list_tables", {"reasoning": reasoning})
    if not DB_PATH:
        error_msg = "Database path not set."
        log_error(error_msg)
        return f"Error: {error_msg}"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        result_str = ", ".join(tables)
        log_function_result("list_tables", result_str)
        return result_str
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"


def tool_describe_table(reasoning: str, table_name: str) -> str:
    """
    Returns schema information about the specified table as a multi-line string.
    """
    log_function_call("describe_table", {"reasoning": reasoning, "table_name": table_name})
    if not DB_PATH:
        error_msg = "Database path not set."
        log_error(error_msg)
        return f"Error: {error_msg}"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        conn.close()
        # Format into a readable multi-line string
        header = "| CID | Name | Type | NotNull | Default | PK |"
        separator = "|---|---|---|---|---|---|"
        rows_str = ["| " + " | ".join(map(str, (c[0], c[1], c[2], bool(c[3]), c[4], bool(c[5])))) + " |" for c in columns]
        result_str = "\n".join([header, separator] + rows_str)
        log_function_result("describe_table", result_str)
        return result_str
    except Exception as e:
        error_msg = f"Error describing table {table_name}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"


def tool_sample_table(reasoning: str, table_name: str, row_sample_size: int) -> str:
    """
    Returns a sample of rows from the specified table as a multi-line string.
    """
    log_function_call("sample_table", {"reasoning": reasoning, "table_name": table_name, "row_sample_size": row_sample_size})
    if not DB_PATH:
        error_msg = "Database path not set."
        log_error(error_msg)
        return f"Error: {error_msg}"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {row_sample_size};")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No rows found in sample."
        # Format as a multi-line string
        headers = list(rows[0].keys())
        rows_str = [" | ".join(map(str, row)) for row in rows]
        result_str = " | ".join(headers) + "\n" + "\n".join(rows_str)
        log_function_result("sample_table", result_str)
        return result_str
    except Exception as e:
        error_msg = f"Error sampling table {table_name}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"


def tool_run_test_sql_query(reasoning: str, sql_query: str) -> str:
    """
    Executes a test SQL query and returns results as a multi-line string.
    """
    log_function_call("run_test_sql_query", {"reasoning": reasoning, "sql_query": sql_query})
    if not DB_PATH:
        error_msg = "Database path not set."
        log_error(error_msg)
        return f"Error: {error_msg}"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.commit()
        conn.close()
        if not rows:
            return "Query executed successfully, no results returned."
        # Format as a multi-line string
        headers = list(rows[0].keys())
        rows_str = [" | ".join(map(str, row)) for row in rows]
        result_str = " | ".join(headers) + "\n" + "\n".join(rows_str)
        log_function_result("run_test_sql_query", result_str)
        return result_str
    except Exception as e:
        error_msg = f"Error running test query '{sql_query}': {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"


def tool_run_final_sql_query(reasoning: str, sql_query: str) -> str:
    """
    Executes the final SQL query and returns results as a multi-line string.
    """
    log_function_call("run_final_sql_query", {"reasoning": reasoning, "sql_query": sql_query})
    if not DB_PATH:
        error_msg = "Database path not set."
        log_error(error_msg)
        # Still return error string, main loop checks for "Error:" prefix
        return f"Error: {error_msg}"
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.commit()
        conn.close()
        if not rows:
            result_str = "Query executed successfully, no results returned."
        else:
            # Format as a multi-line string
            headers = list(rows[0].keys())
            rows_str = [" | ".join(map(str, row)) for row in rows]
            result_str = " | ".join(headers) + "\n" + "\n".join(rows_str)
        log_function_result("run_final_sql_query", result_str)
        # Add a prefix to signal completion to the main loop
        return f"FINAL_RESULT:\n{result_str}"
    except Exception as e:
        error_msg = f"Error running final query '{sql_query}': {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        # Return error string, main loop checks for "Error:" prefix
        return f"Error: {error_msg}"


# Map function names to actual functions
available_functions = {
    "list_tables": tool_list_tables,
    "describe_table": tool_describe_table,
    "sample_table": tool_sample_table,
    "run_test_sql_query": tool_run_test_sql_query,
    "run_final_sql_query": tool_run_final_sql_query,
}

# --- Main Execution Logic ---
def main():
    """Main function to run the SQLite Gemini agent."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="SQLite Agent using Google Gemini API"
    )
    parser.add_argument(
        "-d", "--db", required=True, help="Path to SQLite database file"
    )
    parser.add_argument("-p", "--prompt", required=True, help="The user's request")
    parser.add_argument(
        "-c", "--compute", type=int, default=10, help="Maximum compute loops"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.0-flash", # Use the requested flash model
        help="Gemini model to use (e.g., gemini-2.0-flash, gemini-1.5-pro-latest)",
    )

    args = parser.parse_args()

    # Set global DB_PATH for tool functions
    global DB_PATH
    DB_PATH = args.db
    if not os.path.exists(DB_PATH):
        console.print(f"[red]Error: Database file not found at {DB_PATH}[/red]")
        sys.exit(1)

    # Initialize Gemini client
    # Ensure GEMINI_API_KEY environment variable is set via load_dotenv() or system env
    try:
        # Check if the API key is available after loading .env
        if not os.getenv("GEMINI_API_KEY"):
            console.print("[red]Error: GEMINI_API_KEY environment variable not set.[/red]")
            console.print("Please obtain a key from Google AI Studio and set the environment variable (e.g., in a .env file).")
            sys.exit(1)

        # Initialize the Client.
        # The google-genai SDK automatically uses the GEMINI_API_KEY environment variable.
        client = genai.Client()
    except Exception as e:
        # Catch potential issues during client initialization
        console.print(f"[red]Error initializing Gemini client: {e}[/red]")
        sys.exit(1)

    # Format the initial prompt
    formatted_prompt = AGENT_PROMPT.replace("{{user_request}}", args.prompt)

    # Initialize conversation history (list of Content objects)
    messages: List[genai_types.Content] = [genai_types.Content(role="user", parts=[genai_types.Part(text=formatted_prompt)])]

    # Track number of iterations
    iterations = 0
    max_iterations = args.compute
    break_loop = False

    while iterations < max_iterations:
        if break_loop:
            break

        iterations += 1
        console.rule(f"[yellow]Agent Loop {iterations}/{max_iterations}[/yellow]")

        try:
            # Send message to Gemini using client.models
            response = client.models.generate_content(
                model=args.model, # Specify the model here
                contents=messages,
                # Pass tools via GenerateContentConfig, matching the bash agent pattern
                config=genai_types.GenerateContentConfig(tools=[gemini_tools])
            )

            # Extract the response candidate
            if not response.candidates:
                 console.print("[red]Error: No candidates received from Gemini. Response might be blocked or empty.[/red]")
                 console.print(f"Prompt Feedback: {response.prompt_feedback}")
                 messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text="[Model response was empty or blocked]")]))
                 continue

            candidate = response.candidates[0]

            # Handle safety ratings and finish reasons
            # Correctly check against expected FinishReason enum members (using FUNCTION_CALLS)
            if candidate.finish_reason != genai_types.FinishReason.STOP and candidate.finish_reason != genai_types.FinishReason.FUNCTION_CALLS:
                 console.print(f"[yellow]Warning: Unexpected finish reason: {candidate.finish_reason}[/yellow]")
                 if candidate.safety_ratings:
                     console.print(f"Safety Ratings: {candidate.safety_ratings}")
                 messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text=f"[Model stopped unexpectedly: {candidate.finish_reason}]")]))
                 if candidate.finish_reason == genai_types.FinishReason.MAX_TOKENS:
                     console.print("[yellow]Max tokens reached. Consider increasing model output limit or simplifying the task.[/yellow]")
                     break_loop = True
                 # Continue for other reasons like SAFETY, RECITATION, OTHER for now

            if not candidate.content or not candidate.content.parts:
                console.print("[yellow]Warning: Received empty or incomplete response candidate content/parts.[/yellow]")
                messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text="[Model response content was empty]")]))
                continue

            # --- Process Response Parts ---
            assistant_response_parts = []
            function_calls_to_process = []

            for part in candidate.content.parts:
                if part.text:
                    console.print(Panel(part.text, title="[magenta]Assistant[/magenta]"))
                    assistant_response_parts.append(part)
                elif part.function_call:
                    fc = part.function_call
                    function_name = fc.name
                    try:
                        function_args = dict(fc.args)
                    except Exception as e:
                         log_error(f"Error converting function call args for {function_name}: {e}")
                         function_args = {}

                    console.print(
                        Panel(
                            f"Tool Call Requested: {function_name}({json.dumps(function_args)})",
                            title="[yellow]Tool Call[/yellow]",
                            border_style="yellow",
                        )
                    )
                    function_calls_to_process.append(part)
                else:
                    console.print(f"[yellow]Warning: Received unknown part type: {part}[/yellow]")
                    assistant_response_parts.append(part)

            # Filter out empty text parts before adding the assistant's response to history.
            # Keep function calls and non-empty text parts.
            valid_parts_for_history = [
                part for part in candidate.content.parts
                if part.function_call or (part.text and part.text.strip())
            ]
            if valid_parts_for_history:
                 messages.append(genai_types.Content(role=candidate.content.role, parts=valid_parts_for_history))
            elif not function_calls_to_process: # Log if there was nothing useful at all
                 console.print("[yellow]Warning: Assistant response contained no valid text or function calls after filtering.[/yellow]")


            # --- Execute Function Calls ---
            if function_calls_to_process:
                function_response_parts = []

                for func_call_part in function_calls_to_process:
                    fc = func_call_part.function_call
                    function_name = fc.name
                    try:
                        function_args = dict(fc.args)
                    except Exception as e:
                         log_error(f"Error converting function call args for {function_name} during execution: {e}")
                         function_args = {}

                    if function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        try:
                            # Tool function now returns a simple string or error string
                            function_response_content_str = function_to_call(**function_args)

                            # Check if the task is complete (signaled by 'FINAL_RESULT:' prefix)
                            if function_name == "run_final_sql_query":
                                console.print("\n[bold green]Final Query Executed.[/bold green]")
                                if function_response_content_str.startswith("FINAL_RESULT:"):
                                    final_result_display = function_response_content_str.split("FINAL_RESULT:\n", 1)[1]
                                    console.print("[bold green]Result:[/bold green]")
                                    console.print(final_result_display) # Print the formatted string result
                                    break_loop = True # Task successful
                                elif function_response_content_str.startswith("Error:"):
                                     console.print(f"[red]Final query failed:[/red] {function_response_content_str}")
                                     break_loop = True # Break on final query error
                                else:
                                     # Should not happen if tool adheres to contract
                                     console.print("[yellow]Final query tool finished but response format unexpected.[/yellow]")
                                     break_loop = True

                            # Prepare the function response part for Gemini, wrapping the string result
                            # Use the Part.from_function_response helper
                            function_response_parts.append(
                                genai_types.Part.from_function_response(
                                    name=function_name,
                                    response={"result": function_response_content_str} # Wrap the string result
                                )
                            )

                        except Exception as e:
                            error_msg = f"Error executing function '{function_name}' locally: {str(e)}"
                            log_error(error_msg)
                            console.log(traceback.format_exc())
                            # Ensure error response is a dict containing the error message string
                            error_response_dict = {"error": error_msg}
                            function_response_parts.append(
                                genai_types.Part.from_function_response(
                                    name=function_name,
                                    response=error_response_dict
                                )
                            )
                    else:
                        log_error(f"Unknown function call requested: {function_name}")
                        # Ensure error response is a dict
                        unknown_func_error_dict = {"error": f"Unknown function: {function_name}"}
                        function_response_parts.append(
                             genai_types.Part.from_function_response(
                                name=function_name,
                                response=unknown_func_error_dict
                            )
                        )

                # Add all function responses to history as a single 'function' role message
                if function_response_parts:
                    messages.append(
                        genai_types.Content(role="function", parts=function_response_parts)
                    )
            # Check if the model stopped naturally without requesting more tools
            elif candidate.finish_reason == genai_types.FinishReason.STOP:
                 # Check if there were any valid parts added to history in this turn.
                 # If the only parts were empty text, it might indicate an issue.
                 last_message_parts = messages[-1].parts if messages and messages[-1].role == candidate.content.role else []
                 if not any(p.text.strip() for p in last_message_parts if p.text):
                     console.print("[yellow]Warning: Model stopped but the last response seems empty or only contained non-text parts.[/yellow]")
                 else:
                    console.print("[cyan]Model finished generating (STOP reason) and no further tools requested. Assuming task completion.[/cyan]")
                    break_loop = True
            # This handles the case where there were no function calls *and* no valid text parts were added earlier
            elif not valid_parts_for_history:
                 console.print("[yellow]Warning: Assistant response had neither valid text nor function calls after filtering.[/yellow]")


        except Exception as e:
            log_error(f"An error occurred during the agent loop: {str(e)}")
            console.print(traceback.format_exc())
            # Optionally print history for debugging
            # ... (history printing logic from bash agent) ...
            break # Stop the loop on general errors

    if iterations >= max_iterations and not break_loop:
        log_error("Reached maximum number of iterations without completing the task.")

    console.print("[bold green]Agent finished.[/bold green]")


if __name__ == "__main__":
    main()
