from agents.router_agent import RouterAgent
from agents.langchain_agent import ClaudeLangchainAgent
from tools.spelling_bee_tool import find_valid_word
from tools.simple_negotiation_tools import simple_negotiation_tools
from system_prompts import SPELLINGBEE_PROMPT, STANDARD_GAME_PROMPT, SIMPLE_NEGOTIATION_PROMPT


spellingbee_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SPELLINGBEE_PROMPT, tools=[find_valid_word])
general_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=STANDARD_GAME_PROMPT, tools=[])
simplenegotiation_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SIMPLE_NEGOTIATION_PROMPT, tools=simple_negotiation_tools)

models = [
    {
        'name': 'general_agent',
        'description': 'An agent for general purpose',
        'model': general_agent
    }, 
    {
        'name': 'spellingbee_agent', 
        'description': 'An agent specialized for spellingbee',
        'model': spellingbee_agent
    },
    {   
        'name': 'simplenegotiation_agent',
        'description': 'An agent specialized for simple negotiation',
        'model': simplenegotiation_agent
    }
]

agent = RouterAgent(
    model_name="claude-3-5-sonnet-latest",
    models=models,
)

input_msg = "\n[GAME] You are Player 1 in the Spelling Bee Game.\nAllowed Letters: eilnptv\nEach word must be at least as long as the previous word.\nRepeated words are not allowed.\nWrap your word in square brackets, e.g., '[example]'.\n\n[Player 0] [eline]\n[GAME] Player 0 attempted an invalid move. Reason: Player 0 tried submitting a non-english word. Please resubmit a valid move and remember to follow the game rules to avoid penalties.\n[Player 0] : [pent]"

print(agent(input_msg))