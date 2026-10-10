from pathlib import Path

import yaml

from src.util.llm_client import LLMClient


CONFIG_DIR = Path(__file__).resolve().parent
PUBLIC_CONFIG_PATH = CONFIG_DIR / "config.yml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "config.local.yml"


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_llm_client(profile_name: str) -> LLMClient:
    """Build an LLM client from public settings and local credentials."""
    public_profiles = _read_yaml(PUBLIC_CONFIG_PATH).get("llm_profiles", {})

    if not LOCAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Missing src/config/config.local.yml. Copy "
            "src/config/config.local.example.yml to config.local.yml "
            "and add a newly rotated API key."
        )

    local_profiles = _read_yaml(LOCAL_CONFIG_PATH).get("llm_profiles", {})
    public_profile = public_profiles.get(profile_name)
    local_profile = local_profiles.get(profile_name, {})

    if public_profile is None:
        available = ", ".join(sorted(public_profiles))
        raise KeyError(
            f"Unknown LLM profile '{profile_name}'. Available profiles: {available}"
        )

    api_key = local_profile.get("api_key", "").strip()
    if not api_key or api_key.startswith("replace-with-"):
        raise ValueError(
            f"A valid api_key is required for profile '{profile_name}' "
            "in src/config/config.local.yml."
        )

    return LLMClient(
        api_key=api_key,
        base_url=public_profile["base_url"],
        model=public_profile["model"],
        temperature=public_profile.get("temperature", 0),
    )

