from e2b_code_interpreter import Sandbox


def execute_code_in_sandbox(code_snippet, timeout=30, verbose=True):
    """
    Execute provided code snippet in an E2B Sandbox environment.
    
    Args:
        code_snippet (str): The code to execute
        timeout (int): Maximum execution time in seconds
        verbose (bool): Whether to print additional information
        
    Returns:
        dict: Execution results containing:
            - success (bool): Whether execution completed successfully
            - result (str): Output text from the execution
            - error (str): Error message if execution failed
    """
    if not code_snippet or not code_snippet.strip():
        return {
            "success": False,
            "result": "",
            "error": "No code provided for execution"
        }
    
    try:
        if verbose:
            print(f"Executing code in sandbox (timeout: {timeout}s)...")
        
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code_snippet, timeout=timeout)
            result = execution.text
            
        if verbose:
            print("Code execution completed successfully")
            
        return {
            "success": True,
            "result": result,
            "error": ""
        }
    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        if verbose:
            print(error_message)
        return {
            "success": False,
            "result": "",
            "error": error_message
        }

