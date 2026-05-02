# Modelo juez local usando Ollama como backend
# Ubicación: ~/Documents/DeepEval/judge_model.py

from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaJudge(DeepEvalBaseLLM):
    """
    Modelo juez local usando Ollama.
    Por defecto usa qwen2.5 pero puede cambiarse
    por cualquier modelo disponible en: ollama list
    """

    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama no requiere API key real
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # 0 = respuestas deterministas
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # Versión asíncrona requerida por DeepEval
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Ollama/{self.model_name}"

# — Prueba rápida del juez —
# Ejecuta: python judge_model.py 
