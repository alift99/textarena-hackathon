from langchain.tools import tool
from typing import Dict, Any, Callable, List, Tuple
import random
from typing import TypedDict
import numpy as np
from scipy.optimize import linprog

class Trade(TypedDict):
    offer: Dict[str, int]  # Mapping of resource name to quantity
    request: Dict[str, int]  # Mapping of resource name to quantity

@tool
def resource_evaluator(trade: Trade, accepted: bool) -> str:
    """Estimates opponent resource values based on trade acceptance history.
    
    - If the trade was accepted, it suggests the opponent valued the requested resources **less** than the offered ones.
    - If denied, it suggests the opponent values their requested resources **more** than what was offered.
    """
    opponent_values: Dict[str, float] = {}  # Store estimated values
    
    offered, requested = trade['offer'], trade['request']
    
    for res, qty in requested.items():
        if accepted:
            opponent_values[res] = max(opponent_values.get(res, 10), 0.8 * qty)
        else:
            opponent_values[res] = min(opponent_values.get(res, 40), 1.2 * qty)

    for res, qty in offered.items():
        if accepted:
            opponent_values[res] = min(opponent_values.get(res, 40), 1.2 * qty)  # Opponent values this less
        else:
            opponent_values[res] = max(opponent_values.get(res, 10), 0.8 * qty)  # Opponent values this more

    return f"Updated opponent value estimations: {opponent_values}"

simple_negotiation_tools = [
    resource_evaluator
]
