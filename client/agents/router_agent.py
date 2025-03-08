import os
import time
from typing import Optional, List
import re

from textarena.core import Agent
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from system_prompts import ROUTER_AGENT_PROMPT


def extract_first_bracketed(text):
    match = re.search(r'\[(.*?)\]', text)
    return match.group(1) if match else None


class RouterAgent(Agent):
    """Agent class using the Anthropic API with LangChain tool_calling_agent to generate responses."""
    
    def __init__(
        self, 
        model_name: str, 
        models: List,
        system_prompt: Optional[str] = ROUTER_AGENT_PROMPT,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the Claude agent.
        
        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs

        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        
        self.model = ChatAnthropic(model=model_name)
        models_string = [f"{model['name']}: {model['description']}" for model in models]
        self.system_prompt = system_prompt + f"\nAvailable agents: {models_string}"
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}")
            ]
        )
        self.models = {item["name"]: item["model"] for item in models}
        
    
    def _make_request(self, observation: str) -> str:
        """
        Make a single API request to OpenAI and return the generated message.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response text.
        """

        # Format model input
        messages = [
            ("system", self.system_prompt),
            ("human", observation),
        ]

        # Make the API call using the provided model and messages.
        response = self.model.invoke(messages)

        # Extract the name of the selected agent
        selected_agent_name = extract_first_bracketed(response.content)
        print(f'Selected agent: {selected_agent_name}')

        # Get agent/model
        selected_agent = self.models.get(selected_agent_name)

        # Pass observation to selected agent and return output
        return selected_agent(observation)
    

    def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.
        
        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.
        
        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception
    

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenAI API and return the generated response.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)
