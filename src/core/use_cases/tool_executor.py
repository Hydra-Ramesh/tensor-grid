from typing import Callable, Dict, Awaitable, Optional
from src.engine.tools import get_current_weather, calculate_math
from src.utils.logger import logger


class ToolRegistry:
    """Strategy pattern for dynamic tool execution (Agentic Loop)."""

    def __init__(self):
        self._tools: Dict[str, Callable[[str], Awaitable[str]]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        # In a real FAANG system, we would parse JSON tool calls from the LLM.
        # For this architecture, we register static mock tools.
        self.register("weather", get_current_weather)
        self.register("calculate", calculate_math)

    def register(self, name: str, func: Callable[[str], Awaitable[str]]):
        self._tools[name] = func

    async def execute_tool_for_prompt(self, prompt: str) -> Optional[str]:
        """
        Dynamically selects and executes a tool based on the prompt.
        Returns the augmented prompt if a tool was used, else None.
        """
        prompt_lower = prompt.lower()
        try:
            if "weather" in prompt_lower:
                observation = await self._tools["weather"]("San Francisco")
                return f"System: You have access to a weather tool which returned the following observation:\n{observation}\n\nUser: {prompt}"
            elif (
                "calculate" in prompt_lower
                or "+" in prompt_lower
                or "*" in prompt_lower
            ):
                observation = await self._tools["calculate"]("25 * 4")
                return f"System: You have access to a calculator which returned:\n{observation}\n\nUser: {prompt}"
        except Exception as e:
            logger.error({"event": "tool_execution_failed", "error": str(e)})

        return None
