import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI as OpenAIClient

# ── Modelo juez local ───────────────────────────────────────────
class OllamaJuez(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        response = self.client.chat.completions.create(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt)

    def get_model_name(self):
        return "qwen2.5:3b"

juez = OllamaJuez()

# ── Métricas ────────────────────────────────────────────────────
coherencia = GEval(
    name="Coherencia Pedagógica",
    criteria="Evalúa si las preguntas son claras, relevantes al tema y tienen opciones plausibles de selección múltiple",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=juez
)

relevancia = GEval(
    name="Relevancia del Tema",
    criteria="Evalúa si el contenido generado es relevante y apropiado para el tema solicitado",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=juez
)

calidad = GEval(
    name="Calidad de Rúbricas",
    criteria="Evalúa si las rúbricas de evaluación son útiles, claras y permiten al profesor calificar objetivamente",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=juez
)

# ── Función generadora ──────────────────────────────────────────
client = OpenAIClient(base_url="http://localhost:11434/v1", api_key="ollama")

def generar_respuesta(tema, nivel, modelo):
    prompt = (
        f"Eres un profesor universitario. Genera 3 preguntas teóricas y conceptuales "
        f"de selección múltiple de nivel {nivel} sobre: {tema}. "
        f"Solo texto plano, sin fórmulas ni LaTeX.\n\n"
        f"Formato de respuesta:\n\n"
        f"Pregunta 1: [enunciado completo de la pregunta]\n"
        f"a) [opción a]\n"
        f"b) [opción b]\n"
        f"c) [opción c]\n"
        f"d) [opción d]\n"
        f"Respuesta correcta: [letra]\n"
        f"Rúbrica: [criterio de evaluación]\n\n"
        f"Pregunta 2: [enunciado completo de la pregunta]\n"
        f"a) [opción a]\n"
        f"b) [opción b]\n"
        f"c) [opción c]\n"
        f"d) [opción d]\n"
        f"Respuesta correcta: [letra]\n"
        f"Rúbrica: [criterio de evaluación]\n\n"
        f"Pregunta 3: [enunciado completo de la pregunta]\n"
        f"a) [opción a]\n"
        f"b) [opción b]\n"
        f"c) [opción c]\n"
        f"d) [opción d]\n"
        f"Respuesta correcta: [letra]\n"
        f"Rúbrica: [criterio de evaluación]"
    )
    response = client.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ── Casos de prueba ─────────────────────────────────────────────
casos = [
    ("Programación orientada a objetos", "Básico",  "qwen2.5:3b"),
    ("Programación orientada a objetos", "Alto",    "qwen2.5:3b"),
    ("Redes de computadores",            "Básico",  "qwen2.5:3b"),
    ("Redes de computadores",            "Medio",   "qwen2.5:3b"),
    ("Fundamentos de electrónica",       "Básico",  "qwen2.5:3b"),
    ("Sistemas operativos",              "Medio",   "qwen2.5:3b"),
    ("Inteligencia artificial",          "Básico",  "qwen2.5:3b"),
    ("Programación orientada a objetos", "Básico",  "qwen2.5:1.5b"),
    ("Programación orientada a objetos", "Alto",    "qwen2.5:1.5b"),
    ("Redes de computadores",            "Medio",   "qwen2.5:1.5b"),
]

# ── Tests ───────────────────────────────────────────────────────
@pytest.mark.parametrize("tema,nivel,modelo", casos)
def test_coherencia(tema, nivel, modelo):
    input_text = f"Genera 3 preguntas de nivel {nivel} sobre {tema} usando {modelo}"
    output = generar_respuesta(tema, nivel, modelo)
    test_case = LLMTestCase(
        input=input_text,
        actual_output=output,
        context=[f"Tema: {tema}", f"Nivel: {nivel}", f"Modelo: {modelo}"]
    )
    assert_test(test_case, [coherencia])

@pytest.mark.parametrize("tema,nivel,modelo", casos[:5])
def test_relevancia(tema, nivel, modelo):
    input_text = f"Genera 3 preguntas de nivel {nivel} sobre {tema} usando {modelo}"
    output = generar_respuesta(tema, nivel, modelo)
    test_case = LLMTestCase(
        input=input_text,
        actual_output=output,
        context=[f"Tema: {tema}", f"Nivel: {nivel}", f"Modelo: {modelo}"]
    )
    assert_test(test_case, [relevancia])

@pytest.mark.parametrize("tema,nivel,modelo", casos[:5])
def test_calidad_rubricas(tema, nivel, modelo):
    input_text = f"Genera 3 preguntas de nivel {nivel} sobre {tema} usando {modelo}"
    output = generar_respuesta(tema, nivel, modelo)
    test_case = LLMTestCase(
        input=input_text,
        actual_output=output,
        context=[f"Tema: {tema}", f"Nivel: {nivel}", f"Modelo: {modelo}"]
    )
    assert_test(test_case, [calidad])