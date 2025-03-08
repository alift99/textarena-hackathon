import textarena as ta
from agents.langchain_agent import ClaudeLangchainAgent, OpenAILangchainAgent
from agents.router_agent import RouterAgent
from system_prompts import SPELLINGBEE_PROMPT, STANDARD_GAME_PROMPT
from tools.spelling_bee_tool import find_valid_word

general_agent = ClaudeLangchainAgent(
    model_name="claude-3-5-sonnet-latest", system_prompt=STANDARD_GAME_PROMPT, tools=[]
)
spellingbee_agent = ClaudeLangchainAgent(
    model_name="claude-3-5-sonnet-latest",
    system_prompt=SPELLINGBEE_PROMPT,
    tools=[find_valid_word],
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
]

agent = RouterAgent(
    model_name="claude-3-5-sonnet-latest",
    models=models,
)

# Initialize agents
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    0: agent,
    1: general_agent,
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SimpleNegotiation-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Claude Agent", 1: "Claude"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
