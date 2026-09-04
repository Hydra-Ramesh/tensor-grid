from src.utils.logger import logger


async def get_current_weather(location: str) -> str:
    """Mock external API call to fetch weather data."""
    logger.info(
        {"event": "tool_execution", "tool": "get_current_weather", "location": location}
    )
    # In a real app, this would be an actual HTTPX call to OpenWeatherMap
    return f"The current weather in {location} is 72°F and sunny."


async def calculate_math(expression: str) -> str:
    """Mock external tool to execute math safely."""
    logger.info(
        {"event": "tool_execution", "tool": "calculate_math", "expression": expression}
    )
    try:
        # DO NOT DO THIS IN PRODUCTION WITHOUT A SANDBOX
        # This is purely for demonstrating Agentic tool use architecture
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"
