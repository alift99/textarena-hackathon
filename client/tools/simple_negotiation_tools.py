from langchain.tools import tool
from typing import Dict, Any, Callable, List, Tuple
import random
from typing import TypedDict
import numpy as np
from scipy.optimize import linprog
import re

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

@tool
def compute_offer_value(offer: str, value: Dict[str, int]) -> int:
    """
    Computes the value of a trade offer.

    The offer must be in the format: '[Offer: <your resources> -> <their resources>]'
    Example: '[Offer: 2 Wheat, 1 Ore -> 3 Sheep]'

    Args:
        offer (str): The trade offer string.
        value (Dict[str, int]): A dictionary mapping resource names to their values.

    Returns:
        int: The net value gained from the trade (positive means profit, negative means loss).
    """
    pattern = r"^\[Offer: (.*?) -> (.*?)\]$"
    match = re.match(pattern, offer)

    if not match:
        raise ValueError("Invalid offer format. Expected '[Offer: <your resources> -> <their resources>]'.")
    
    your_resources = parse_resources(match.group(1))
    their_resources = parse_resources(match.group(2))

    your_value = sum(value[res] * amount for res, amount in your_resources.items() if res in value)
    their_value = sum(value[res] * amount for res, amount in their_resources.items() if res in value)
    print(f"{your_value - their_value}")
    return their_value - your_value

def parse_resources(resource_str: str) -> Dict[str, int]:
    """
    Parses a resource string into a dictionary.

    Example: "2 Wheat, 1 Ore" -> {"Wheat": 2, "Ore": 1}
    """
    resources = {}
    if resource_str.strip():
        items = resource_str.split(',')
        for item in items:
            parts = item.strip().split(' ', 1)
            if len(parts) == 2:
                amount, resource = parts
                resources[resource] = int(amount)
    return resources



simple_negotiation_tools = [
    resource_evaluator, 
    compute_offer_value
]
