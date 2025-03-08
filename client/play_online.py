import textarena as ta

from agents.langchain_agent import OpenAILangchainAgent, ClaudeLangchainAgent
from tools.spelling_bee_tool import find_valid_word
from system_prompts import SPELLINGBEE_PROMPT
 

model_name = "test agent"
model_description = "test agent"
email = "alifdaffa.main@gmail.com"


# Initialize agent
agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SPELLINGBEE_PROMPT, tools=[find_valid_word]) 


env = ta.make_online(
    env_id=["SpellingBee-v0"], 
    model_name=model_name,
    model_description=model_description,
    email=email
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)


rewards = env.close()