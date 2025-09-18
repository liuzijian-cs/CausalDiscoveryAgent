# src/llm/LLMClient.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-18 20:20
# 引用: https://github.com/Lancelot39/Causal-Copilot
# 描述: LLM客户端封装器 (LLM Client Wrapper)

import os
import re
import json
import logging
from typing import Optional, Union, Dict, Any

from openai import OpenAI


class PydanticHandler:
    """处理Pydantic模型相关逻辑 (Handle Pydantic model related logic)"""
    
    @staticmethod
    def generate_schema_instruction(pydantic_model):
        """为结构化输出生成JSON schema指令 (Generate JSON schema instruction for structured output)"""
        if not hasattr(pydantic_model, 'model_json_schema'):
            return "请以有效的JSON格式响应。"
        
        try:
            schema = pydantic_model.model_json_schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            instruction = "请以JSON格式响应:\n{"
            for field_name, field_info in properties.items():
                field_type = field_info.get('type', 'any')
                is_required = field_name in required
                req_text = "必需" if is_required else "可选"
                instruction += f'\n  "{field_name}": {field_type} ({req_text})'
            instruction += "\n}"
            return instruction
        except Exception:
            return "请以有效的JSON格式响应。"
    
    @staticmethod
    def parse_to_model(data: dict, pydantic_model):
        """尝试将字典解析为Pydantic模型 (Try to parse dictionary to Pydantic model)"""
        try:
            return pydantic_model.model_validate(data)
        except Exception:
            try:
                return pydantic_model(**data)
            except Exception as e:
                raise ValueError(f"无法解析数据到模型 {pydantic_model.__name__}: {e}")


class LLMClient(object):
    def __init__(self, 
                 provider: str = "default",
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None
                 ):
        """
        初始化LLM客户端 (Initialize LLM client)
        
        Args:
            provider (str): 提供商类型，目前仅支持"default" (Provider type, currently only supports "default")
            model (str): 使用的模型名称 (Model name to use)
            api_key (Optional[str]): API密钥，如果未提供则从环境变量LLM_API_KEY获取 (API key, gets from LLM_API_KEY if not provided)
            base_url (Optional[str]): API端点，如果未提供则从环境变量LLM_API_URL获取 (API endpoint, gets from LLM_API_URL if not provided)
        """
        self.logger = logging.getLogger(__name__)
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # 验证提供商类型 (Validate provider type)
        assert provider in ["default"], "目前仅支持'default'提供商 (Currently only 'default' provider is supported)"
        self.provider = provider
        self.model = model
    
        # 从环境变量或参数获取API密钥 (Get API key from environment variable or parameter)
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
        assert api_key is not None, "必须通过参数或LLM_API_KEY环境变量提供API密钥 (API key must be provided via parameter or LLM_API_KEY environment variable)"
        self.api_key = api_key

        # 从环境变量或参数获取API端点 (Get API endpoint from environment variable or parameter)
        if base_url is None:
            base_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1")
        assert base_url is not None, "必须通过参数或LLM_API_URL环境变量提供API端点 (API endpoint must be provided via parameter or LLM_API_URL environment variable)"
        self.base_url = base_url
        
        # 初始化OpenAI客户端 (Initialize OpenAI client)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
    def _ensure_json_in_messages(self, messages):
        """确保消息中包含'json'关键字以满足OpenAI JSON模式要求 (Ensure messages contain 'json' keyword for OpenAI JSON mode)"""
        json_mentioned = any('json' in msg.get('content', '').lower() for msg in messages)
        if not json_mentioned:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += " 请以JSON格式响应。"
            else:
                messages.insert(0, {"role": "system", "content": "请以JSON格式响应。"})
        return messages
    
    def chat_completion(self, 
                       prompt: Optional[str] = None,
                       system_prompt: str = "你是一个有用的助手。", 
                       messages: Optional[list] = None,
                       response_format = None,
                       json_response: bool = False, 
                       model: Optional[str] = None,
                       temperature: float = 0.3) -> Union[str, Dict, Any]:
        """
        向LLM发送聊天完成请求 (Send a chat completion request to LLM)
        
        Args:
            prompt (str, optional): 用户提示词 (User prompt)
            system_prompt (str): 系统提示词 (System prompt)  
            messages (list, optional): OpenAI格式的消息列表 (List of messages in OpenAI format)
            response_format (optional): 用于结构化输出的Pydantic模型 (Pydantic model for structured output)
            json_response (bool): 是否期望JSON响应 (Whether to expect JSON response)
            model (str, optional): 覆盖默认模型 (Override the default model)
            temperature (float): 响应生成的温度参数 (Temperature for response generation)
            
        Returns:
            Union[str, Dict, Any]: 模型的响应 (The model's response)
        """
        # 构建消息 (Build messages)
        if messages is None:
            if prompt is None:
                raise ValueError("必须提供messages或prompt参数 (Either messages or prompt must be provided)")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        # 处理结构化输出 (Handle structured output)
        if response_format is not None:
            schema_instruction = PydanticHandler.generate_schema_instruction(response_format)
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{schema_instruction}"
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            json_response = True
        
        # 获取响应 (Get response)
        if json_response:
            messages = self._ensure_json_in_messages(messages)
        
        # 构建请求参数 (Build request parameters)
        request_params = {
            "model": model if model else self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # 添加响应格式参数 (Add response format parameter)
        if json_response:
            request_params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        
        # 解析响应 (Parse response)
        if json_response or response_format is not None:
            if content is None:
                raise ValueError("收到空的响应内容 (Received empty response content)")
                
            try:
                parsed_json = json.loads(content) if isinstance(content, str) else content
                
                if response_format is not None:
                    return PydanticHandler.parse_to_model(parsed_json, response_format)
                return parsed_json
                
            except json.JSONDecodeError:
                # 尝试从代码块中提取JSON (Try to extract JSON from code blocks)
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(1))
                        if response_format is not None:
                            return PydanticHandler.parse_to_model(parsed_json, response_format)
                        return parsed_json
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f"解析JSON响应失败 (Failed to parse JSON response): {content}")
        
        return content 

if __name__ == "__main__":
    """
    测试LLM客户端 (Test LLM client)
    """
    from dotenv import load_dotenv
    load_dotenv()
    test_llm_client = LLMClient(
        provider="default",
        model=os.getenv("MODEL", "gpt-5"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("LLM_API_URL")
    )
    test_response = test_llm_client.chat_completion(prompt="你好！请介绍一下你自己！",
                                                    system_prompt="你是一个因果分析助手！你需要辅助用户进行因果发现任务！",
                                                    temperature=1.0,
                                                    json_response=True,
                                                    )
    print(test_response)