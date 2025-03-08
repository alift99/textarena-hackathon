from functools import reduce
from langchain_core.tools import tool

@tool
def solve_nim(stacks_input):
    """
    this is a tool that solves a Nim game position and returns the optimal move.
    input a string of space-separated integers representing stack sizes
    
    Args:
        stacks_input (str): A string of space-separated integers representing stack sizes
                           Example: "1 3 5 7"
    
    Returns:
        dict: A dictionary containing the solution information:
              - 'winning': True if there's a winning move, False otherwise
              - 'move': Dictionary with 'stack_index' and 'items_to_remove' if winning
              - 'message': Human-readable explanation of the result
    """
    # Parse input
    stack = stacks_input.strip().split()
    if not stack:
        return {"winning": False, "message": "No stacks provided"}
    
    int_stack = list(map(int, stack))
    nim_sum = reduce(lambda x, y: x^y, int_stack)
    
    # If nim_sum is 0, current position is losing
    if not nim_sum:
        return {
            "winning": False,
            "message": "No winning move available - current position is losing"
        }
    
    # Calculate optimal move
    min_stack = list(map(lambda x: x^nim_sum if x-(x^nim_sum) >= 0 else 100000, int_stack))
    min_value = min(min_stack)
    min_index = min_stack.index(min_value)
    to_drop_items = (int_stack[min_index] - min_value) or 1
    
    return {
        "winning": True,
        "move": {
            "stack_index": min_index,  # 0-indexed
            "stack_number": min_index + 1,  # 1-indexed for human readability
            "items_to_remove": to_drop_items
        },
        "message": f"Remove {to_drop_items} item(s) from stack {min_index+1}"
    }


# Example usage (can be removed if not needed)
if __name__ == "__main__":
    print("Nim Game Solver")
    print("Enter stacks separated by space.")
    print("Example: 1 3 5 7")
    
    while True:
        stacks_input = input("Enter stacks (or press Enter to quit): ")
        if not stacks_input:
            print('Finished!')
            break
        
        result = solve_nim(stacks_input)
        print(result["message"])