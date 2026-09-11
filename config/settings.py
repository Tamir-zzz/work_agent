"""全局配置：通过环境变量或 .env 文件加载。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM 配置 (OpenAI 兼容接口)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-xxxx"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # 记忆配置
    memory_db_path: Path = PROJECT_ROOT / "data" / "memory.db"
    # 长期记忆容量与遗忘
    memory_capacity: int = 200          # 超过该量触发 LRU 遗忘裁剪
    # 冲突去重阈值：新记忆与已有记忆相似度超过该值则合并而非新增
    memory_conflict_threshold: float = 0.85

    # Embedding 配置 (OpenAI 兼容 /v1/embeddings)
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 16

    # Agent 循环配置
    max_plan_depth: int = 3        # 任务树最大深度
    max_agent_steps: int = 10      # 单轮最大思考-行动步数


settings = Settings()