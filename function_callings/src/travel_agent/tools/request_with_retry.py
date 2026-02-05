import time
import requests

def request_with_retry(url, params=None, max_retry=3):
    retry = 0
    success = False
    while not success and retry <= max_retry:
        response = requests.get(url, params)
        data = response.json()
        status = data['status']
        info = data['info']
        if status == '1':
            break
        else:
            retry += 1
            print(f"request failed, status={status}, info={info}, retry={retry}, max_retry={max_retry}, url={url}, param={params}.")
            time.sleep(retry)
    return data

