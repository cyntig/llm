# !pip install openai python-dotenv tavily-python requests
from dotenv import load_dotenv
import requests
import os
import json
from openai import OpenAI

load_dotenv()

def get_current_weather(city): 
    """获取城市天气"""
    amap_api_key = os.environ["AMAP_API_KEY"]

    # 获取行政区域编码
    url = f"https://restapi.amap.com/v3/config/district?key={amap_api_key}&keywords={city}"
    response = requests.get(url)
    acode = response.json().get("districts")[0]["adcode"]
    
    # 获取天气信息
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={amap_api_key}&city={acode}&extensions=base"
    response = requests.get(url)
    weather_json = response.json().get("lives")[0]

    return json.dumps(weather_json, ensure_ascii=False, indent=4)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "A city name like Beijing or Shanghai"
                    }
                }
            }
        }
    }
]

def chat_weather(model, fn_map): 
    messages = [{"role": "user", "content": "北京、上海天气怎么样"}]
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice='auto'
    )
    resp.model_dump()
    tool_calls = resp.choices[0].message.tool_calls
    tool_call_params = [{
        "role": "assistant",
        "tool_calls": [tool_call.model_dump() for tool_call in tool_calls]
    }]
    tool_call_result = [
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": fn_map[tool_call.function.name](**json.loads(tool_call.function.arguments))
    }
    for tool_call in tool_calls]

    resp_reply = client.chat.completions.create(
        model=model,
        messages=messages + tool_call_params + tool_call_result,
        tools=tools,
        tool_choice='auto', 
        temperature=0
    )
    print(resp_reply.choices[0].message.content)

if __name__ == "__main__":
    model = "Qwen/Qwen3-8B"
    fn_map = {
        'get_current_weather': get_current_weather
    }
    chat_weather(model, fn_map)
