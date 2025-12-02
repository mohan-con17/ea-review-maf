from pydantic_settings  import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # <--- THIS is the important part
    )

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str
    
    # Azure AI Search
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_INDEX_NAME: str
    AZURE_AI_SEARCH_SERVICE_NAME: str
    AZURE_SEARCH_KEY: str

    # Azure Blob Storage
    AZURE_BLOB_CONNECTION_STRING: str
    AZURE_CONTAINER_NAME: str
    AZURE_LOGS_CONTAINER_NAME: str
    AZURE_ACCOUNT_NAME: str
    AZURE_ACCOUNT_KEY: str

    # Local paths
    chat_history_dir: Path = Path(__file__).resolve().parents[1] / "chat-history"

    # class Config:
    #     env_file = ".env"

settings = Settings()
