import textarena as ta
from agents.langchain_agent import ClaudeLangchainAgent, ClaudeLangchainAgentCustom
from agents.router_agent import RouterAgent
from system_prompts import (
    SIMPLE_NEGOTIATION_PROMPT,
    SPELLINGBEE_PROMPT,
    STANDARD_GAME_PROMPT,
)
from tools.simple_negotiation_tools import simple_negotiation_tools
from tools.spelling_bee_tool import find_valid_word

model_name = "HumanGPT-negotiation"
model_description = "A mixture of customized agents, orchestrated by a *router* agent"
email = "simshanghong@gmail.com"

general_agent = ClaudeLangchainAgent(
    model_name="claude-3-5-sonnet-latest", system_prompt=STANDARD_GAME_PROMPT, tools=[]
)
spellingbee_agent = ClaudeLangchainAgent(
    model_name="claude-3-5-sonnet-latest",
    system_prompt=SPELLINGBEE_PROMPT,
    tools=[find_valid_word],
)
simplenegotiation_agent = ClaudeLangchainAgentCustom(
    model_name="claude-3-5-sonnet-latest",
    system_prompt=SIMPLE_NEGOTIATION_PROMPT,
    tools=simple_negotiation_tools,
)

models = [
    {
        "name": "general_agent",
        "description": "An agent for general purpose",
        "model": general_agent,
    },
    {
        "name": "spellingbee_agent",
        "description": "An agent specialized for spellingbee",
        "model": spellingbee_agent,
    },
    {
        "name": "simplenegotiation_agent",
        "description": "An agent specialized for simple negotiation",
        "model": simplenegotiation_agent,
    },
]

agent = RouterAgent(
    model_name="claude-3-5-sonnet-latest",
    models=models,
)

# Initialize environment from subset and wrap it
env = ta.make_online(
    env_id="SimpleNegotiation-v0",
    model_name=model_name,
    model_description=model_description,
    email=email,
)
env = ta.wrappers.LLMObservationWrapper(env=env)

env.reset(num_players=1)
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)
rewards = env.close()
