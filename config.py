"""
config.py — Configuration loader for Manny RAG Chatbot v4
"""

import os
import yaml


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "agent.yaml")

        raw      = _load_yaml(yaml_path)
        base_dir = os.path.dirname(os.path.abspath(yaml_path))

        # ── Agent ─────────────────────────────────────────────────────────
        self.agent_name    = raw["agent"]["name"]
        self.agent_version = raw["agent"]["version"]

        # ── Shop ──────────────────────────────────────────────────────────
        shop = raw["shop"]
        self.shop_name  = shop["name"]
        self.shop_city  = shop["city"]
        self.shop_phone = shop["phone"]
        self.shop_hours = shop["hours"]

        # ── Model (Groq) ──────────────────────────────────────────────────
        model = raw["model"]
        self.model_provider = model["provider"]
        self.model_name     = model["name"]
        self.vision_model   = model["vision_name"]
        self.model_timeout  = model.get("timeout", 30)
        self.max_tokens     = model.get("max_tokens", 1024)

        # ── Embeddings ────────────────────────────────────────────────────
        self.embed_model = raw["embeddings"]["model"]

        # ── RAG ───────────────────────────────────────────────────────────
        rag = raw["rag"]
        self.knowledge_dir   = os.path.join(base_dir, rag["knowledge_dir"])
        self.collection_name = rag["collection_name"]
        self.chunk_size      = rag["chunk_size"]
        self.chunk_overlap   = rag["chunk_overlap"]
        self.retriever_k     = rag["retriever_k"]

        # ── Vision ────────────────────────────────────────────────────────
        vision = raw.get("vision", {})
        self.vision_enabled = vision.get("enabled", False)
        self.vision_prompt  = vision.get("prompt", "Describe what you see.")
        self.vision_formats = vision.get("accepted_formats",
                                         ["image/jpeg", "image/png", "image/webp"])
        self.vision_max_mb  = vision.get("max_size_mb", 10)

        # ── Diagnostic flow ───────────────────────────────────────────────
        self.diagnostic_questions = raw["diagnostic_questions"]
        self.symptom_keywords     = raw["symptom_keywords"]

        # ── Customer prompts ──────────────────────────────────────────────
        subs = {
            "agent_name": self.agent_name,
            "shop_name":  self.shop_name,
            "shop_city":  self.shop_city,
            "shop_phone": self.shop_phone,
            "shop_hours": self.shop_hours,
        }
        self.prompt_base      = raw["prompts"]["base"].format(**subs)
        self.prompt_diagnosis = raw["prompts"]["diagnosis"].format(**subs)

        # ── Mechanic prompts ──────────────────────────────────────────────
        mech = raw.get("mechanic_prompts", {})
        self.mechanic_prompt_base      = mech.get("base", self.prompt_base).format(**subs)
        self.mechanic_prompt_diagnosis = mech.get("diagnosis", self.prompt_diagnosis).format(**subs)

        # ── Car model lookup ──────────────────────────────────────────────
        cml = raw.get("car_model_lookup", {})
        self.car_lookup_enabled = cml.get("enabled", False)
        self.car_lookup_num     = cml.get("num_problems", 5)
        self.car_lookup_brands  = [b.lower() for b in cml.get("brand_keywords", [])]
        _car_prompt = cml.get("prompt", "List {num_problems} common problems for {car_model}.")
        self.car_lookup_prompt  = _car_prompt.format(
            agent_name   = self.agent_name,
            shop_name    = self.shop_name,
            car_model    = "{car_model}",
            num_problems = "{num_problems}",
        )

    def __repr__(self):
        return (
            f"Config(agent={self.agent_name} v{self.agent_version}, "
            f"model={self.model_name}, vision={self.vision_model}, "
            f"embed={self.embed_model})"
        )
