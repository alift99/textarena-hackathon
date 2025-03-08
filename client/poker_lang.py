from phevaluator import evaluate_cards

import os
from dotenv import load_dotenv

from test import poker_odds
import textarena as ta

from client.langchain_agent import OpenAILangchainAgent, ClaudeLangchainAgent
from langchain_core.tools import tool
# Load API key from .env
load_dotenv()

# Initialize agentsfrom langchain.agents.agent_types import AgentType
POKER_PROMPT = "You are a competitive POKER player. ALWAYS USE THE TOOL Your strategy is to leverage your tool to check how strong your cards are. you can use it to think about what you want to do next. be careful the opponent might think that we are too rational and use that against us"
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    0: ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest",system_prompt= POKER_PROMPT,tools=[poker_odds]),
    1: ta.agents.AnthropicAgent(model_name="claude-3-7-sonnet-latest"),
}

# Initialize environment from subset and wrap it
env = ta.make   (env_id="Poker-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Player 1", 1: "Player 2"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()



# def gto_poker_agent(observation):
#     """
#     This is a GTO poker agent that uses the phevaluator library to evaluate the cards.unset HTTP_PROXY HTTPS_PROXY
#     """
#     cards = observation.split("Cards:")[1].split("Pot:")[0].strip()
#     eval = evaluate_cards(cards)
#     print(cards)
#     return cards


# @tool
# def evaluate_cards(mycards):
#     """
#     you can use this to compute the rank of your cards
#     takes my cards and opponent cards and returns the rank of the cards
#     Convert these poker cards to evaluator format:
#     - Replace card symbols with letters: ♣→c, ♦→d, ♥→h, ♠→s
#     - Replace 10 with T
#     - Replace face cards: Jack→J, Queen→Q, King→K, Ace→A
#     - Format should be rank+suit with no spaces (e.g., "Qh", "Tc", "As")

#     Examples:
#     "Q♥" → "Qh"
#     "10♣" → "Tc"
#     "Ace of spades" → "As"
#     "K♦, 7♥" → "Kd, 7h"
        
#     """
#     eval1 = evaluate_cards(mycards)
#     print(eval1)
    
#     string_output = f"My cards are ranked {eval1} according to phevaluator"
#     print(string_output)
#     return string_output