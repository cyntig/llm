#!/usr/bin/python
# -*- coding: UTF-8 -*-

from openai import OpenAI
import sys
import os


class ChatOpenAI: 
    def __init__(self, base_url=os.environ.get('OPENAI_BASE_URL'), api_key=os.environ['OPENAI_API_KEY']):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def build_messages(self, sys_prompt = None, user_prompt = None):
        msgs = []
        if sys_prompt is not None:
            msgs.append({"role": "system", "content": sys_prompt})
        
        if user_prompt is not None:
            msgs.append({"role": "user", "content": user_prompt})
        return msgs
    
    def chat_completions(self, model_name, messages, max_tokens = 10, temperature = 0.1, **kwargs): 
        try:
            response = self.client.chat.completions.create(
                model = model_name,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature,
                **kwargs
            )
            sentiment = response.choices[0].message.content.strip()
            return sentiment
        except Exception as e:
            print(f"OpenAI调用出错: {e}")
            return None
        
if __name__ == "__main__":
    llm = ChatOpenAI()
    print(llm.build_messages(sys_prompt = "abc", user_prompt = "bcd"))