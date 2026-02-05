from dotenv import load_dotenv
import requests
import os
import json

import travel_agent.tools.request_with_retry as request_with_retry

load_dotenv()

name = "get_foods"

def get_foods_for_page(location, page, radius=10000, rating=4.5):
    """获取POI附近美食"""
    amap_api_key = os.environ["AMAP_API_KEY"]
    url = f"https://restapi.amap.com/v3/place/around?"
    params = {
        'key': amap_api_key,
        'location': location, 
        'types': "中餐厅",
        'radius': radius,
        'sorted': "weight",
        'offset': 10,
        'page': page
    }
    data = request_with_retry.request_with_retry(url, params)
    if data == None:
        return "未查到相关餐厅"
    else:
        pois = data['pois']

        # 取评分超过4.5的
        good_foods = [{
            'name': poi['name'],
            'type': poi['type'],
            'address': poi['address'],
            'cityname': poi['cityname'],
            'tel': poi['tel'],
            'location': poi['location'],
            'distance': poi['distance'],
            'rating': poi['biz_ext']['rating'],
            'cost': poi['biz_ext']['cost']
        } for poi in pois if poi['biz_ext'].get('rating') and float(poi['biz_ext']['rating']) >= rating]
        return good_foods

def get_foods(location):
    all_foods = []
    for page in range(0, 4): 
        all_foods.extend(get_foods_for_page(location, page))
    return json.dumps(all_foods, ensure_ascii=False, indent=4)

get_foods_schema = {
    "type": "function",
    "function": {
        "name": "get_foods",
        "description": "Get the foods of a sepcial location",
        "parameters": {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location is longitude and latitude, like '100.164000,25.694836'."
                },
            }
        }
    }
}

if __name__ == "__main__":
    print(get_foods('100.164000,25.694836'))