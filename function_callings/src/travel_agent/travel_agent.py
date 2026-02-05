# !pip install openai python-dotenv tavily-python requests
from dotenv import load_dotenv
import requests
import os
import json
from openai import OpenAI

from travel_agent.tools import get_famous_spots
from travel_agent.tools import get_future_weather
from travel_agent.tools import get_foods

load_dotenv()

tools = [
    get_famous_spots.get_famous_spots_schema, 
    get_future_weather.get_future_weather_schema,
    get_foods.get_foods_schema
]

fn_map = {
    "get_famous_spots": get_famous_spots.get_famous_spots,
    "get_future_weather": get_future_weather.get_future_weather,
    "get_foods": get_foods.get_foods
}

def chat_travel_plan(user_question):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )

    messages = [] 
    messages.append({
        "role": "system",
        "content": f"""
            你是一个专业的旅行规划师，请根据用户输入的地点和时间制定一份合理的旅行计划，包括每天行程、美食、天气等信息，你可以这样做：
            - 第一步，规划每天游玩的城市和景点
            - 第二步，提供游玩景点所在城市的天气
            - 第三步，推荐3个游玩景点附近美食
        """
    })
    messages.append ({
        "role": "user",
        "content": user_question
    })

    finish_reason = None
    round_cnt = 0
    while finish_reason is None or finish_reason == "tool_calls":
        round_cnt += 1
        resp = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages, 
            tools=tools, 
            tool_choice="auto"
        )
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call.model_dump() for tool_call in resp.choices[0].message.tool_calls]
            })
            print(f"Round {round_cnt}: {choice.message.tool_calls}")
            for tool_call in choice.message.tool_calls:
                print(tool_call.model_dump())
                tool_call_name = tool_call.function.name
                tool_call_arguments = tool_call.function.arguments
                tool_call_result = fn_map[tool_call_name](**json.loads(tool_call_arguments))
                print(tool_call_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_call_result
                    }
                )
    return resp.choices[0].message.content
    
    

if __name__ == "__main__":
    user_question = "请帮我规划一个2026年2月5日至2月7日的云南旅游，需要具体到城市、天、每天天气"
    print(chat_travel_plan(user_question)) 