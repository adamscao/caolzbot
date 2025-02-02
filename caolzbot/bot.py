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
        self.conversation_histories = {}
        
    async def generate_response(self, prompt: str, user_id: str) -> str:
        raise NotImplementedError
        
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

class OpenAIClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        if not api_key:
            self.logger.error("OpenAI API密钥未设置")

    async def generate_response(self, prompt: str, user_id: str) -> str:
        # 预处理prompt，移除@caolzbot
        prompt = prompt.replace("caolzbot", "").strip()
        
        history = self._get_or_create_history(user_id)
        history.append({"role": "user", "content": prompt})
        
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

    async def generate_response(self, prompt: str, user_id: str) -> str:
        # 预处理prompt，移除@caolzbot
        prompt = prompt.replace("caolzbot", "").strip()
        
        history = self._get_or_create_history(user_id)
        history.append({"role": "user", "content": prompt})
        
        if len(history) > 20:
            history = history[-20:]
            
        # Anthropic需要特殊处理历史消息格式
        messages = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
            
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": messages
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
                        response_content = result["content"][0]["text"]
                        history.append({"role": "assistant", "content": response_content})
                        return response_content
                    else:
                        self.logger.error(f"Anthropic请求失败: {response.status} - {response_text}")
                        return f"错误: {response.status} - {response_text}"
            except Exception as e:
                self.logger.exception("Anthropic请求过程中发生异常")
                return f"请求异常: {str(e)}"

class DeepseekClient(AIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        if not api_key:
            self.logger.error("Deepseek API密钥未设置")

    async def generate_response(self, prompt: str, user_id: str) -> str:
        # 预处理prompt，移除@caolzbot
        prompt = prompt.replace("caolzbot", "").strip()
        
        history = self._get_or_create_history(user_id)
        
        # 如果是新对话，添加system message
        if not history:
            history.append({"role": "system", "content": "You are a helpful assistant"})
            
        history.append({"role": "user", "content": prompt})
        
        if len(history) > 20:
            # 保留system message
            system_msg = history[0] if history[0]["role"] == "system" else None
            history = history[-20:]
            if system_msg and history[0]["role"] != "system":
                history.insert(0, system_msg)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": history,
            "stream": False
        }
        
        self.logger.debug(f"发送请求到Deepseek - Headers: {headers}")
        self.logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.endpoint, headers=headers, json=data) as response:
                    response_text = await response.text()
                    self.logger.debug(f"Deepseek响应状态码: {response.status}")
                    self.logger.debug(f"Deepseek响应内容: {response_text}")
                    
                    if response.status == 401:
                        self.logger.error("Deepseek认证失败，请检查API密钥")
                        return "认证失败，请检查API密钥设置"
                    elif response.status == 200:
                        result = json.loads(response_text)
                        assistant_message = result["choices"][0]["message"]
                        history.append(assistant_message)
                        return assistant_message["content"]
                    else:
                        self.logger.error(f"Deepseek请求失败: {response.status} - {response_text}")
                        return f"错误: {response.status} - {response_text}"
            except Exception as e:
                self.logger.exception("Deepseek请求过程中发生异常")
                return f"请求异常: {str(e)}"

class MultiAIBot(ActivityHandler):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ai_clients = {}
        self.user_ai_choices = {}  # 新增：存储每个用户的AI选择
        
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
            
        # 初始化Deepseek客户端
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            self.ai_clients["deepseek"] = DeepseekClient(deepseek_key)
        else:
            self.logger.error("未找到DEEPSEEK_API_KEY环境变量")
            
        self.default_ai = "openai" if "openai" in self.ai_clients else next(iter(self.ai_clients), None)
        
        if not self.ai_clients:
            self.logger.critical("没有可用的AI客户端，请检查API密钥设置")

    def _get_user_ai_choice(self, user_id: str) -> str:
        """获取用户当前选择的AI，如果没有则返回默认AI"""
        return self.user_ai_choices.get(user_id, self.default_ai)

    async def on_message_activity(self, turn_context: TurnContext):
        self.logger.info(f"收到用户消息: {turn_context.activity.text}")
        # 预处理消息，移除@前缀
        message = turn_context.activity.text.lower()
        message = message.replace("caolzbot", "").strip()
        self.logger.debug(f"处理后的消息: {message}")
        
        user_id = turn_context.activity.from_property.id
        
        # 获取当前用户的AI选择
        ai_choice = self._get_user_ai_choice(user_id)

        # 命令处理
        if message.startswith("$"):
            command = message.split()[0]
            
            # $list命令
            if command == "$list":
                available_ais = ", ".join(self.ai_clients.keys())
                current_ai = self._get_user_ai_choice(user_id)
                await turn_context.send_activity(
                    f"可用的AI模型: {available_ais}\n"
                    f"当前使用的AI: {current_ai}"
                )
                return

            # $clear命令
            elif command == "$clear":
                self.logger.info(f"用户 {user_id} 请求清除对话历史")
                if ai_choice in self.ai_clients:
                    response = self.ai_clients[ai_choice].clear_history(user_id)
                    await turn_context.send_activity(response)
                return

            # $use命令
            elif command == "$use":
                parts = message.split()
                if len(parts) > 1 and parts[1] in self.ai_clients:
                    self.user_ai_choices[user_id] = parts[1]
                    self.logger.info(f"用户切换AI到: {parts[1]}")
                    await turn_context.send_activity(f"已切换至 {parts[1]}")
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
                    "支持的命令：\n"
                    "$list - 列出所有可用的AI模型\n"
                    "$use <ai名称> - 切换到指定的AI模型\n"
                    "$clear - 清除当前对话历史\n"
                )
                
                if self.ai_clients:
                    welcome_message += f"当前可用的AI模型: {', '.join(self.ai_clients.keys())}"
                else:
                    welcome_message += "警告：当前没有可用的AI模型，请联系管理员配置"
                
                self.logger.info(f"发送欢迎消息给新用户: {member.id}")
                await turn_context.send_activity(welcome_message)