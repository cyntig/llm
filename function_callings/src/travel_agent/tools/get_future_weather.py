from dotenv import load_dotenv
import requests
import os
import json

import travel_agent.tools.request_with_retry as request_with_retry

load_dotenv()

name = "get_future_weather"

def get_adcode(city): 
    """获取行政区域编码adcode"""
    amap_api_key = os.environ["AMAP_API_KEY"]

    # 获取行政区域编码
    url = f"https://restapi.amap.com/v3/config/district?key={amap_api_key}&keywords={city}"
    reponse_json = request_with_retry.request_with_retry(url, 3)
    if reponse_json['status'] == '1': 
        return reponse_json.get('districts')[0]['adcode']
    else:
        return None

def _get_future_weather_use_acode(acode, date):
    amap_api_key = os.environ["AMAP_API_KEY"]
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={amap_api_key}&city={acode}&extensions=all"
    response_json = request_with_retry.request_with_retry(url)
    if response_json['status'] != '1':
        return "未查到天气情况"
    else:
        weather_json = response_json.get("forecasts")[0]
        weather_json["casts"] = [cast for cast in weather_json["casts"] if cast["date"] == date]
        if weather_json["casts"] == []:
            weather_json["casts"] = "未查到天气情况"
        return json.dumps(weather_json, ensure_ascii=False, indent=4)


def get_future_weather(city, date): 
    url = ""
    """获取城市天气"""
    amap_api_key = os.environ["AMAP_API_KEY"]

    acode = get_adcode(city)
    if acode == None:
        return "未查到天气情况"
    else:
       return _get_future_weather_use_acode(acode, date)

get_future_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_future_weather",
        "description": "Get the weather of a city for a particular day",
        "parameters": {
            "type": "object",
            "required": ["city", "date"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "A city name like 北京 or 上海"
                },
                "date": {
                    "type": "string",
                    "description": "The target future date, formatted as “yyyy-MM-dd, like 2026-02-02"
                }
            }
        }
    }
}

if __name__ == "__main__":
    print(get_future_weather("北京", "2026-02-06"))