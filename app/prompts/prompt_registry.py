# import yaml
# from pathlib import Path

# class PromptRegistry:
#     _cache = {}

#     @classmethod
#     def load(cls):
#         base = Path("app/prompts/registry")
#         for file in base.rglob("*.yaml"):
#             data = yaml.safe_load(file.read_text())
#             key = f"{data['prompt_id']}:{data['version']}"
#             cls._cache[key] = data

#     @classmethod
#     def get(cls, prompt_id: str, version: str):
#         key = f"{prompt_id}:{version}"
#         if key not in cls._cache:
#             raise KeyError(f"Prompt not found: {key}")
#         return cls._cache[key]

import yaml
from pathlib import Path

class PromptRegistry:
    _cache = {}

    @classmethod
    def load(cls):
        base = Path("app/prompts/registry")

        for file in base.rglob("*.yaml"):
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise RuntimeError(
                    f"Invalid encoding in prompt file: {file}. "
                    "All prompt YAML files must be UTF-8 encoded."
                ) from e

            data = yaml.safe_load(text)

            if not data or "prompt_id" not in data or "version" not in data:
                raise ValueError(
                    f"Invalid prompt definition in {file}. "
                    "Missing required fields: prompt_id/version."
                )

            key = f"{data['prompt_id']}:{data['version']}"
            cls._cache[key] = data

    @classmethod
    def get(cls, prompt_id: str, version: str):
        key = f"{prompt_id}:{version}"
        if key not in cls._cache:
            raise KeyError(f"Prompt not found: {key}")
        return cls._cache[key]
