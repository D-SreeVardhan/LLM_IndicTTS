"""Central configuration – loaded once at startup."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pydantic v2

ROOT = Path(__file__).resolve().parents[2]  # LLM_Prj/

load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    # ── HuggingFace ──
    hf_token: str = Field("", env="HF_TOKEN")
    hf_home: str = Field("", env="HF_HOME")

    # ── Model IDs ──
    asr_model_id: str = Field(
        "ai4bharat/indic-conformer-600m-multilingual", env="ASR_MODEL_ID"
    )
    trans_indic_en_model_id: str = Field(
        "ai4bharat/indictrans2-indic-en-dist-200M", env="TRANS_INDIC_EN_MODEL_ID"
    )
    trans_en_indic_model_id: str = Field(
        "ai4bharat/indictrans2-en-indic-dist-200M", env="TRANS_EN_INDIC_MODEL_ID"
    )
    llm_model_id: str = Field(
        "bharatgenai/Param-1-2.9B-Instruct", env="LLM_MODEL_ID"
    )
    tts_model_id: str = Field(
        "ai4bharat/indic-parler-tts", env="TTS_MODEL_ID"
    )
    embed_model_id: str = Field("BAAI/bge-m3", env="EMBED_MODEL_ID")

    # ── Retrieval ──
    rag_top_k: int = Field(5, env="RAG_TOP_K")
    rag_min_score: float = Field(0.25, env="RAG_MIN_SCORE")

    # ── Runtime ──
    keep_llm_resident: bool = Field(True, env="KEEP_LLM_RESIDENT")
    keep_embed_resident: bool = Field(True, env="KEEP_EMBED_RESIDENT")
    keep_tts_resident: bool = Field(True, env="KEEP_TTS_RESIDENT")
    asr_decode_mode: Literal["ctc", "rnnt"] = Field("rnnt", env="ASR_DECODE_MODE")
    hindi_route: Literal["direct", "pivot", "both"] = Field("both", env="HINDI_ROUTE")

    # ── API ──
    api_host: str = Field("127.0.0.1", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    cors_origins: str = Field("http://localhost:5173", env="CORS_ORIGINS")

    # ── Paths (derived) ──
    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed"

    @property
    def index_dir(self) -> Path:
        return ROOT / "data" / "processed" / "index"

    class Config:
        env_file = str(ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
