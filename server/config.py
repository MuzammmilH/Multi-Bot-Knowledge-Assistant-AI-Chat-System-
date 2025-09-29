from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    INDEX_DIR: str = "./indexes"
    FAISS_INDEX_PATH: str = "./indexes/faiss.index"
    METADATA_PATH: str = "./indexes/metadata.pkl"

    class Config:
        env_file = ".env"

settings = Settings()
