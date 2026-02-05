from dotenv import load_dotenv
import requests
import os
import json

load_dotenv()

from travel_agent.tools import request_with_retry

name = "get_famous_spots"

def _get_famous_spots_for_page(city, page, rating):
    """获取城市热门景点"""
    amap_api_key = os.environ["AMAP_API_KEY"]
    
    # POI 分类编码：风景名胜
    # 参考：https://lbs.amap.com/api/webservice/download
    types = "风景名胜"
    params = {
        "key": amap_api_key,
        "types": "风景名胜",
        "city": city,
        "citylimit": True, 
        "page": page,
        "offset": 20,
        "extension": "all"
    }
    url = f"https://restapi.amap.com/v3/place/text"

    data = request_with_retry.request_with_retry(url, params) 

    if data == None:
        return "未查到热门景点"
    else:
        pois = data['pois']

        # 取评分超过4.5的
        famous_pois = [{
            'name': poi['name'],
            'address': poi['address'],
            'cityname': poi['cityname'],
            'tel': poi['tel'],
            'location': poi['location'],
            'rating': poi['biz_ext']['rating'],
            'cost': poi['biz_ext']['cost']
        } for poi in pois if poi['biz_ext'].get('rating') and float(poi['biz_ext']['rating']) >= rating]

        return famous_pois


def get_famous_spots(city, rating=4.5): 
    all_pois = []
    for page in range(0, 1): 
        all_pois.extend(_get_famous_spots_for_page(city, page, rating))
    return json.dumps(all_pois, ensure_ascii=False, indent=4)

get_famous_spots_schema = {
    "type": "function",
    "function": {
        "name": "get_famous_spots",
        "description": "Get the famous spots of a sepcial region",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "A city name like 云南"
                },
            }
        }
    }
}

if __name__ == "__main__":
    print(get_famous_spots("北京"))



     


