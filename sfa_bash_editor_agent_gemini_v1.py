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
Usage:
    # View a file
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "Show me the first 10 lines of README.md"

    # Create a new file
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "Create a new file called hello_gemini.txt with 'Hello Gemini World!' in it"

    # Replace text in a file
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "Create a new file called hello_gemini.txt with 'Hello Gemini World!' in it. Then update hello_gemini.txt to say 'Hello Gemini AI Coding World'"

    # Insert a line in a file
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "Create a new file called hello_gemini2.txt with 'Hello Gemini AI Coding World!' in it. Then add a new line 'How are you?' after 'Hello Gemini AI Coding World!' in hello_gemini2.txt"

    # Execute a bash command
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "List all Python files in the current directory sorted by size"

    # Complete a multi-step task
    uv run sfa_bash_editor_agent_gemini_v1.py --prompt "List all Python files in the current directory sorted by size, then output to a markdown file called python_files_sorted_by_size_gemini.md"
"""

import os
import sys
import argparse
import json
import traceback
import subprocess
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Global variable for bash environment (can be reset)
current_bash_env = os.environ.copy()

# --- Define Tools using Gemini's FunctionDeclaration ---

view_file_func = genai_types.FunctionDeclaration(
    name="view_file",
    description="View the content of a file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why you are viewing the file",
            ),
            "path": genai_types.Schema(
                type=genai_types.Type.STRING, description="Path of the file to view"
            ),
        },
        required=["reasoning", "path"],
    ),
)

create_file_func = genai_types.FunctionDeclaration(
    name="create_file",
    description="Create a new file with given content",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Why the file is being created",
            ),
            "path": genai_types.Schema(
                type=genai_types.Type.STRING, description="Path where to create the file"
            ),
            "file_text": genai_types.Schema(
                type=genai_types.Type.STRING, description="Content for the new file"
            ),
        },
        required=["reasoning", "path", "file_text"],
    ),
)

str_replace_func = genai_types.FunctionDeclaration(
    name="str_replace",
    description="Replace text in a file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explain why the replacement is needed",
            ),
            "path": genai_types.Schema(
                type=genai_types.Type.STRING, description="File path"
            ),
            "old_str": genai_types.Schema(
                type=genai_types.Type.STRING, description="The string to be replaced"
            ),
            "new_str": genai_types.Schema(
                type=genai_types.Type.STRING, description="The replacement string"
            ),
        },
        required=["reasoning", "path", "old_str", "new_str"],
    ),
)

insert_line_func = genai_types.FunctionDeclaration(
    name="insert_line",
    description="Insert text at a specific line in a file",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Reason for inserting the text",
            ),
            "path": genai_types.Schema(
                type=genai_types.Type.STRING, description="File path"
            ),
            "insert_line": genai_types.Schema(
                type=genai_types.Type.INTEGER, description="Line number for insertion (0-based)"
            ),
            "new_str": genai_types.Schema(
                type=genai_types.Type.STRING, description="The text to insert"
            ),
        },
        required=["reasoning", "path", "insert_line", "new_str"],
    ),
)

execute_bash_func = genai_types.FunctionDeclaration(
    name="execute_bash",
    description="Execute a bash command",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explain why this command should be executed",
            ),
            "command": genai_types.Schema(
                type=genai_types.Type.STRING, description="The bash command to run"
            ),
        },
        required=["reasoning", "command"],
    ),
)

restart_bash_func = genai_types.FunctionDeclaration(
    name="restart_bash",
    description="Restart the bash session with a fresh environment",
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "reasoning": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Explain why the session is being reset",
            )
        },
        required=["reasoning"],
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
        view_file_func,
        create_file_func,
        str_replace_func,
        insert_line_func,
        execute_bash_func,
        restart_bash_func,
        complete_task_func,
    ]
)

# --- Agent Prompt ---
AGENT_PROMPT = """<purpose>
    You are an expert integration assistant using Google Gemini that can both edit files and execute bash commands.
</purpose>

<instructions>
    <instruction>Use the tools provided to accomplish file editing and bash command execution as needed.</instruction>
    <instruction>When you have completed the user's task, call complete_task to finalize the process.</instruction>
    <instruction>Provide reasoning with every tool call.</instruction>
    <instruction>When constructing paths use /repo to start from the root of the repository. We'll replace it with the current working directory.</instruction>
    <instruction>For file editing, prefer using `str_replace` or `insert_line` for targeted changes. Use `create_file` for new files or complete overwrites.</instruction>
    <instruction>If a bash command fails, analyze the error and try to correct it or inform the user.</instruction>
</instructions>

<tools_description>
    - view_file: View the content of a file.
    - create_file: Create a new file or overwrite an existing one.
    - str_replace: Replace a specific string within a file.
    - insert_line: Insert a new line of text at a specific line number (0-based index).
    - execute_bash: Run a bash command in the current environment.
    - restart_bash: Reset the bash environment variables to the system default.
    - complete_task: Signal that the user's request is fully satisfied.
</tools_description>

<user_request>
    {{user_request}}
</user_request>
"""

root_path_to_replace_with_cwd = "/repo"

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

def log_function_result(function_name: str, result: str):
    """Log a function result in a rich panel."""
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
# (Copied and adapted from sfa_bash_editor_agent_anthropic_v3.py)

def tool_view_file(reasoning: str, path: str) -> str:
    """View the content of a file."""
    log_function_call("view_file", {"reasoning": reasoning, "path": path})
    try:
        if path:
            path = path.replace(root_path_to_replace_with_cwd, os.getcwd())

        if not path or not path.strip():
            error_message = "Invalid file path provided: path is empty."
            log_error(error_message)
            return f"Error: {error_message}"

        console.log(f"[tool_view_file] reasoning: {reasoning}, path: {path}")

        if not os.path.exists(path):
            error_message = f"File {path} does not exist"
            log_error(error_message)
            return f"Error: {error_message}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        log_function_result("view_file", f"Read {len(content)} characters from {path}")
        return content
    except Exception as e:
        error_msg = f"Error viewing file {path}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_create_file(reasoning: str, path: str, file_text: str) -> str:
    """Create a new file with given content."""
    log_function_call("create_file", {"reasoning": reasoning, "path": path, "file_text": f"{len(file_text)} chars"})
    try:
        if path:
            path = path.replace(root_path_to_replace_with_cwd, os.getcwd())
        console.log(f"[tool_create_file] reasoning: {reasoning}, path: {path}")

        if not path or not path.strip():
            error_message = "Invalid file path provided: path is empty."
            log_error(error_message)
            return f"Error: {error_message}"

        dirname = os.path.dirname(path)
        if dirname: # Only create dirs if dirname is not empty (i.e., not creating in root)
             os.makedirs(dirname, exist_ok=True)
        elif not dirname and not os.path.exists(os.getcwd()): # Handle case where CWD might not exist? Unlikely but safe.
             os.makedirs(os.getcwd(), exist_ok=True)


        with open(path, "w", encoding="utf-8") as f:
            f.write(file_text or "")
        result_msg = f"File created/updated at {path} with {len(file_text)} characters."
        log_function_result("create_file", result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"Error creating file {path}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_str_replace(reasoning: str, path: str, old_str: str, new_str: str) -> str:
    """Replace text in a file."""
    log_function_call("str_replace", {"reasoning": reasoning, "path": path, "old_str": old_str, "new_str": new_str})
    try:
        if path:
            path = path.replace(root_path_to_replace_with_cwd, os.getcwd())

        if not path or not path.strip():
            error_message = "Invalid file path provided: path is empty."
            log_error(error_message)
            return f"Error: {error_message}"

        if old_str is None: # Allow empty old_str? Let's prevent it for now.
            error_message = "No text to replace specified: old_str is missing or null."
            log_error(error_message)
            return f"Error: {error_message}"

        console.log(
            f"[tool_str_replace] reasoning: {reasoning}, path: {path}, old_str: {old_str}, new_str: {new_str}"
        )

        if not os.path.exists(path):
            error_message = f"File {path} does not exist"
            log_error(error_message)
            return f"Error: {error_message}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str not in content:
            error_message = f"'{old_str}' not found in {path}"
            log_error(error_message)
            # Return error but maybe let the agent decide next step?
            return f"Error: {error_message}"

        new_content = content.replace(old_str, new_str if new_str is not None else "")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        result_msg = f"Text replaced successfully in {path}"
        log_function_result("str_replace", result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"Error replacing text in {path}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_insert_line(reasoning: str, path: str, insert_line: int, new_str: str) -> str:
    """Insert text at a specific line in a file."""
    log_function_call("insert_line", {"reasoning": reasoning, "path": path, "insert_line": insert_line, "new_str": new_str})
    try:
        if path:
            path = path.replace(root_path_to_replace_with_cwd, os.getcwd())

        if not path or not path.strip():
            error_message = "Invalid file path provided: path is empty."
            log_error(error_message)
            return f"Error: {error_message}"

        if insert_line is None:
            error_message = "No line number specified: insert_line is missing."
            log_error(error_message)
            return f"Error: {error_message}"

        # Allow empty string insertion
        # if not new_str:
        #     error_message = "No text to insert specified: new_str is empty."
        #     log_error(error_message)
        #     return f"Error: {error_message}"

        console.log(
            f"[tool_insert_line] reasoning: {reasoning}, path: {path}, insert_line: {insert_line}, new_str: {new_str}"
        )

        if not os.path.exists(path):
            error_message = f"File {path} does not exist"
            log_error(error_message)
            return f"Error: {error_message}"

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust index check for 0-based index
        if not (0 <= insert_line <= len(lines)):
            error_message = (
                f"Insert line number {insert_line} out of range (0-{len(lines)})."
            )
            log_error(error_message)
            return f"Error: {error_message}"

        # Ensure the inserted string ends with a newline if it doesn't already
        line_to_insert = new_str if new_str.endswith('\n') else new_str + '\n'

        lines.insert(insert_line, line_to_insert)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        result_msg = f"Line inserted successfully at index {insert_line} in {path}"
        log_function_result("insert_line", result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"Error inserting line in {path}: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_execute_bash(reasoning: str, command: str) -> str:
    """Execute a bash command."""
    log_function_call("execute_bash", {"reasoning": reasoning, "command": command})
    global current_bash_env
    try:
        if command:
            command = command.replace(root_path_to_replace_with_cwd, os.getcwd())

        if not command or not command.strip():
            error_message = "No command specified: command is empty."
            log_error(error_message)
            return f"Error: {error_message}"

        console.log(f"[tool_execute_bash] reasoning: {reasoning}, command: {command}")

        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, env=current_bash_env, cwd=os.getcwd()
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout.strip()}\n"
        if result.stderr:
             # Treat stderr as part of the result, not necessarily an error for the tool itself
            output += f"STDERR:\n{result.stderr.strip()}\n"

        if result.returncode != 0:
            # Log non-zero exit code but return output (including stderr)
            log_error(f"Command '{command}' exited with code {result.returncode}")
            # Append exit code info to the output returned to the model
            output += f"\nExit Code: {result.returncode}"
            log_function_result("execute_bash", output)
            # Return the combined output, letting the model decide if it's an error state
            return output
        else:
            result_msg = output if output else "Command executed successfully with no output."
            log_function_result("execute_bash", result_msg)
            return result_msg

    except Exception as e:
        error_msg = f"Error executing bash command '{command}': {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_restart_bash(reasoning: str) -> str:
    """Restart the bash session with a fresh environment."""
    log_function_call("restart_bash", {"reasoning": reasoning})
    global current_bash_env
    try:
        if not reasoning:
            error_message = "No reasoning provided for restarting bash session."
            log_error(error_message)
            return f"Error: {error_message}"

        console.log(f"[tool_restart_bash] reasoning: {reasoning}")
        current_bash_env = os.environ.copy()
        result_msg = "Bash session restarted with default environment."
        log_function_result("restart_bash", result_msg)
        return result_msg
    except Exception as e:
        error_msg = f"Error restarting bash session: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        return f"Error: {error_msg}"

def tool_complete_task(reasoning: str) -> str:
    """Finalize the task and exit the agent loop."""
    log_function_call("complete_task", {"reasoning": reasoning})
    try:
        if not reasoning:
            error_message = "No reasoning provided for task completion."
            log_error(error_message)
            # Even with error, signal completion intent
            return "Task completion signaled (missing reasoning)."

        console.log(f"[tool_complete_task] reasoning: {reasoning}")
        result_msg = "Task completed successfully."
        log_function_result("complete_task", result_msg)
        return result_msg # This signals the main loop to break
    except Exception as e:
        error_msg = f"Error during task completion: {str(e)}"
        log_error(error_msg)
        console.log(traceback.format_exc())
        # Still signal completion intent despite logging error
        return f"Task completion signaled (error during finalization: {error_msg})."


# Map function names to actual functions
available_functions = {
    "view_file": tool_view_file,
    "create_file": tool_create_file,
    "str_replace": tool_str_replace,
    "insert_line": tool_insert_line,
    "execute_bash": tool_execute_bash,
    "restart_bash": tool_restart_bash,
    "complete_task": tool_complete_task,
}

# --- Main Execution Logic ---
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Bash and Editor Agent using Google Gemini"
    )
    parser.add_argument("-p", "--prompt", required=True, help="The prompt to execute")
    parser.add_argument(
        "-c", "--compute", type=int, default=10, help="Maximum compute loops"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.0-flash", # Use the requested model
        help="Gemini model to use (e.g., gemini-2.0-flash, gemini-1.5-pro)",
    )

    args = parser.parse_args()

    # Initialize Gemini client
    # Ensure GEMINI_API_KEY environment variable is set
    try:
        client = genai.Client()
    except Exception as e:
        console.print(f"[red]Error initializing Gemini client: {e}[/red]")
        console.print(
            "[yellow]Please ensure the GEMINI_API_KEY environment variable is set.[/yellow]"
        )
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
            # Send message to Gemini
            response = client.models.generate_content(
                model=args.model,
                contents=messages,
                config=genai_types.GenerateContentConfig(tools=[gemini_tools])
            )

            # Extract the response candidate
            # Handle potential errors like blocked responses or empty candidates
            if not response.candidates:
                 console.print("[red]Error: No candidates received from Gemini. Response might be blocked or empty.[/red]")
                 console.print(f"Prompt Feedback: {response.prompt_feedback}")
                 # Add error message to history?
                 messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text="[Model response was empty or blocked]")]))
                 continue # Skip processing this turn

            candidate = response.candidates[0]

            # Handle safety ratings and finish reasons
            if candidate.finish_reason != genai_types.FinishReason.STOP and candidate.finish_reason != genai_types.FinishReason.TOOL_CALLS:
                 console.print(f"[yellow]Warning: Unexpected finish reason: {candidate.finish_reason}[/yellow]")
                 if candidate.safety_ratings:
                     console.print(f"Safety Ratings: {candidate.safety_ratings}")
                 # Add a message indicating the non-standard stop
                 messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text=f"[Model stopped unexpectedly: {candidate.finish_reason}]")]))
                 # Decide whether to break or continue based on the reason
                 if candidate.finish_reason == genai_types.FinishReason.MAX_TOKENS:
                     console.print("[yellow]Max tokens reached. Consider increasing model output limit or simplifying the task.[/yellow]")
                     break_loop = True # Likely can't continue effectively
                 # Continue for other reasons like SAFETY, RECITATION, OTHER for now

            # Ensure content and parts exist before accessing
            if not candidate.content or not candidate.content.parts:
                console.print("[yellow]Warning: Received empty or incomplete response candidate content/parts.[/yellow]")
                # Add placeholder to history
                messages.append(genai_types.Content(role="model", parts=[genai_types.Part(text="[Model response content was empty]")]))
                continue # Skip processing this turn

            # --- Process Response Parts ---
            # Gemini might return multiple parts (e.g., text and function call)
            assistant_response_parts = []
            function_calls_to_process = []

            for part in candidate.content.parts:
                if part.text:
                    console.print(Panel(part.text, title="[magenta]Assistant[/magenta]"))
                    assistant_response_parts.append(part)
                elif part.function_call:
                    fc = part.function_call
                    function_name = fc.name
                    # Convert Struct to dict carefully, handling potential errors
                    try:
                        function_args = dict(fc.args)
                    except Exception as e:
                         log_error(f"Error converting function call args for {function_name}: {e}")
                         function_args = {} # Proceed with empty args? Or handle as error?

                    console.print(
                        Panel(
                            f"Tool Call Requested: {function_name}({json.dumps(function_args)})",
                            title="[yellow]Tool Call[/yellow]",
                            border_style="yellow",
                        )
                    )
                    function_calls_to_process.append(part) # Store the whole part
                else:
                    console.print(f"[yellow]Warning: Received unknown part type: {part}[/yellow]")
                    assistant_response_parts.append(part) # Add unknown parts as is

            # Add assistant's response (text and unprocessed parts) to history
            if assistant_response_parts:
                 messages.append(genai_types.Content(role="model", parts=assistant_response_parts))
            # Add the function call parts separately IF they exist, to match expected history structure for next turn
            if function_calls_to_process:
                 messages.append(genai_types.Content(role="model", parts=function_calls_to_process))


            # --- Execute Function Calls ---
            if function_calls_to_process:
                function_response_parts = [] # Collect responses for the next API call

                for func_call_part in function_calls_to_process:
                    fc = func_call_part.function_call
                    function_name = fc.name
                    try:
                        function_args = dict(fc.args)
                    except Exception as e:
                         log_error(f"Error converting function call args for {function_name} during execution: {e}")
                         function_args = {} # Or skip this call?

                    # Execute the function
                    if function_name in available_functions:
                        function_to_call = available_functions[function_name]
                        try:
                            # Call the function with unpacked arguments
                            function_response_content = function_to_call(**function_args)

                            # Check if the task is complete
                            if function_name == "complete_task":
                                # Check the actual response content in case of errors in the tool itself
                                if "error" not in function_response_content.lower():
                                     break_loop = True
                                else:
                                     console.print("[yellow]Complete_task tool reported an error, not breaking loop.[/yellow]")


                            # Prepare the function response part for Gemini
                            function_response_parts.append(
                                genai_types.Part.from_function_response(
                                    name=function_name,
                                    response={"result": function_response_content}, # Wrap result
                                )
                            )

                        except Exception as e:
                            error_msg = f"Error executing function '{function_name}' locally: {str(e)}"
                            log_error(error_msg)
                            console.log(traceback.format_exc())
                            # Add error response part
                            function_response_parts.append(
                                genai_types.Part.from_function_response(
                                    name=function_name,
                                    response={"error": error_msg},
                                )
                            )
                    else:
                        log_error(f"Unknown function call requested: {function_name}")
                        # Add error response part for unknown function
                        function_response_parts.append(
                            genai_types.Part.from_function_response(
                                name=function_name,
                                response={"error": f"Unknown function: {function_name}"},
                            )
                        )

                # Add all function responses to history as a single 'function' role message
                if function_response_parts:
                    messages.append(
                        genai_types.Content(role="function", parts=function_response_parts)
                    )

            elif not assistant_response_parts:
                # If there was no text and no function call, something might be wrong
                console.print("[yellow]Warning: Assistant response had neither text nor function calls.[/yellow]")
                # Let the loop continue, maybe the model will recover or it was just an empty thought

        except Exception as e:
            log_error(f"An error occurred during the agent loop: {str(e)}")
            console.print(traceback.format_exc())
            console.print("[yellow]Current Conversation History (abbreviated):[/yellow]")
            # Safely print history parts (abbreviated)
            for i, msg in enumerate(messages):
                try:
                    role = msg.role
                    parts_summary = []
                    for part in msg.parts:
                        if part.text:
                            summary = part.text[:100] + ('...' if len(part.text) > 100 else '')
                            parts_summary.append(f"Text: '{summary}'")
                        elif part.function_call:
                            parts_summary.append(f"FunctionCall: {part.function_call.name}")
                        elif part.function_response:
                             parts_summary.append(f"FunctionResponse: {part.function_response.name}")
                        else:
                             parts_summary.append("Other Part")
                    console.print(f"  {i}. Role: {role}, Parts: [{', '.join(parts_summary)}]")
                except Exception as print_e:
                    console.print(f"  [red]Error printing message {i}: {print_e}[/red]")
            break # Stop the loop on general errors

    if iterations >= max_iterations and not break_loop:
        log_error("Reached maximum number of iterations without completing the task.")

    console.print("[bold green]Agent finished.[/bold green]")


if __name__ == "__main__":
    main()
