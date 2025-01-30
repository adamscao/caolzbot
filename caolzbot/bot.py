import logging
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount
import aiohttp
import os
from typing import Dict, Any, List
import json

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIClient:
    """AI模型客户端的基类"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def generate_response(self, prompt: str, user_id: str) -> str:
        raise NotImplementedError
        
    def clear_history(self, user_id: str):
        raise NotImplementedError

class OpenAIClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.conversation_histories = {}  # 用户ID -> 对话历史
        if not api_key:
            self.logger.error("OpenAI API密钥未设置")

    def clear_history(self, user_id: str):
        """清除指定用户的对话历史"""
        if user_id in self.conversation_histories:
            self.conversation_histories[user_id] = []
            self.logger.info(f"已清除用户 {user_id} 的对话历史")
            return "对话历史已清除"
        return "没有找到对话历史"

    def _get_or_create_history(self, user_id: str) -> List[Dict[str, str]]:
        """获取或创建用户的对话历史"""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        return self.conversation_histories[user_id]

    async def generate_response(self, prompt: str, user_id: str) -> str:
        history = self._get_or_create_history(user_id)
        
        # 添加用户的新消息
        history.append({"role": "user", "content": prompt})
        
        # 保持历史记录在合理范围内(最多保留10条消息)
        if len(history) > 20:
            history = history[-20:]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o",
            "messages": history
        }
        
        self.logger.debug(f"发送请求到OpenAI - Headers: {headers}")
        self.logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    response_text = await response.text()
                    self.logger.debug(f"OpenAI响应状态码: {response.status}")
                    self.logger.debug(f"OpenAI响应内容: {response_text}")
                    
                    if response.status == 401:
                        self.logger.error("OpenAI认证失败，请检查API密钥")
                        return "认证失败，请检查API密钥设置"
                    elif response.status == 200:
                        result = json.loads(response_text)
                        assistant_message = result["choices"][0]["message"]
                        # 将助手的回复添加到历史记录
                        history.append(assistant_message)
                        return assistant_message["content"]
                    else:
                        self.logger.error(f"OpenAI请求失败: {response.status} - {response_text}")
                        return f"错误: {response.status} - {response_text}"
            except Exception as e:
                self.logger.exception("OpenAI请求过程中发生异常")
                return f"请求异常: {str(e)}"

class AnthropicClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.anthropic.com/v1/messages"
        if not api_key:
            self.logger.error("Anthropic API密钥未设置")

    async def generate_response(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        self.logger.debug(f"发送请求到Anthropic - Headers: {headers}")
        self.logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    response_text = await response.text()
                    self.logger.debug(f"Anthropic响应状态码: {response.status}")
                    self.logger.debug(f"Anthropic响应内容: {response_text}")
                    
                    if response.status == 401:
                        self.logger.error("Anthropic认证失败，请检查API密钥")
                        return "认证失败，请检查API密钥设置"
                    elif response.status == 200:
                        result = json.loads(response_text)
                        return result["content"][0]["text"]
                    else:
                        self.logger.error(f"Anthropic请求失败: {response.status} - {response_text}")
                        return f"错误: {response.status} - {response_text}"
            except Exception as e:
                self.logger.exception("Anthropic请求过程中发生异常")
                return f"请求异常: {str(e)}"

class MultiAIBot(ActivityHandler):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ai_clients = {}
        
        # 初始化OpenAI客户端
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.ai_clients["openai"] = OpenAIClient(openai_key)
        else:
            self.logger.error("未找到OPENAI_API_KEY环境变量")
            
        # 初始化Anthropic客户端
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.ai_clients["anthropic"] = AnthropicClient(anthropic_key)
        else:
            self.logger.error("未找到ANTHROPIC_API_KEY环境变量")
            
        self.default_ai = "openai" if "openai" in self.ai_clients else next(iter(self.ai_clients), None)
        
        if not self.ai_clients:
            self.logger.critical("没有可用的AI客户端，请检查API密钥设置")

    async def on_message_activity(self, turn_context: TurnContext):
        self.logger.info(f"收到用户消息: {turn_context.activity.text}")
        message = turn_context.activity.text.lower()
        ai_choice = self.default_ai
        user_id = turn_context.activity.from_property.id

        # 检查是否是清除历史的命令
        if message.strip() == "/clear":
            self.logger.info(f"用户 {user_id} 请求清除对话历史")
            if ai_choice in self.ai_clients:
                response = self.ai_clients[ai_choice].clear_history(user_id)
                await turn_context.send_activity(response)
            return

        # 检查是否有切换AI的命令
        if message.startswith("/use "):
            parts = message.split()
            if len(parts) > 1 and parts[1] in self.ai_clients:
                ai_choice = parts[1]
                self.logger.info(f"用户切换AI到: {ai_choice}")
                await turn_context.send_activity(f"已切换至 {ai_choice}")
                return
            else:
                self.logger.warning(f"用户尝试切换到未知的AI: {message}")
                await turn_context.send_activity(f"可用的AI模型: {', '.join(self.ai_clients.keys())}")
                return

        # 生成回复
        try:
            if not self.ai_clients:
                self.logger.error("没有可用的AI客户端")
                await turn_context.send_activity("系统错误：未配置任何AI服务，请联系管理员")
                return
                
            if ai_choice not in self.ai_clients:
                self.logger.error(f"所选AI客户端 {ai_choice} 不可用")
                await turn_context.send_activity("所选AI服务不可用，请选择其他服务或联系管理员")
                return
                
            self.logger.info(f"使用 {ai_choice} 生成响应")
            response = await self.ai_clients[ai_choice].generate_response(message, user_id)
            await turn_context.send_activity(response)
        except Exception as e:
            self.logger.exception("处理消息时发生异常")
            await turn_context.send_activity(f"发生错误: {str(e)}")

    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome_message = (
                    "欢迎！我是一个多AI聊天机器人。\n"
                    "使用'/use <ai名称>'来切换不同的AI模型。\n"
                    "使用'/clear'来清除当前对话历史。\n"
                )
                
                if self.ai_clients:
                    welcome_message += f"当前可用的AI模型: {', '.join(self.ai_clients.keys())}"
                else:
                    welcome_message += "警告：当前没有可用的AI模型，请联系管理员配置"
                
                self.logger.info(f"发送欢迎消息给新用户: {member.id}")
                await turn_context.send_activity(welcome_message)
