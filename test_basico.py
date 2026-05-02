from judge_model import OllamaJudge
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, GEval

# — Instanciar el juez una sola vez para todos los tests —
juez = OllamaJudge(model_name="qwen2.5:3b")

# — Test 1: Relevancia de la respuesta —
# Evalúa si la respuesta es pertinente a la pregunta
def test_relevancia():
    test_case = LLMTestCase(
        input="¿Cuál es la capital de Francia?",
        actual_output="La capital de Francia es París, conocida por la Torre Eiffel."
    )
    metrica = AnswerRelevancyMetric(
        threshold=0.4,
        model=juez,
        verbose_mode=True  # muestra razonamiento del juez
    )
    assert_test(test_case, [metrica])

# — Test 2: Detección de alucinaciones —
# Verifica que la respuesta no invente hechos fuera del contexto
def test_alucinacion():
    test_case = LLMTestCase(
        input="¿Cuántos planetas tiene el sistema solar?",
        actual_output="El sistema solar tiene 8 planetas oficiales.",
        context=["Desde 2006, la IAU reconoce 8 planetas en el sistema solar."]
    )
    metrica = HallucinationMetric(
        threshold=0.5,
        model=juez,
        verbose_mode=True
    )
    assert_test(test_case, [metrica])

# — Test 3: Respuesta incorrecta (debe FALLAR intencionalmente) —
# Sirve para comprobar que DeepEval detecta respuestas falsas
def test_alucinacion_falsa_esperada():
    test_case = LLMTestCase(
        input="¿Cuántos planetas tiene el sistema solar?",
        actual_output="El sistema solar tiene 12 planetas y uno de ellos es Plutón.",
        context=["Desde 2006, la IAU reconoce 8 planetas en el sistema solar."]
    )
    metrica = HallucinationMetric(
        threshold=0.5,
        model=juez,
        verbose_mode=True
    )
    metrica.measure(test_case)
    print(f"⚠️ ESPERADO QUE FALLE Score: {metrica.score} | Reason: {metrica.reason}")
    assert metrica.score > 0.5, "✅ Correcto: el juez detectó la alucinación"

# — Test 4: Criterio personalizado con GEval —
# Define su propio criterio de evaluación en lenguaje natural
def test_geval_personalizado():
    test_case = LLMTestCase(
        input="Explica qué es el overfitting en machine learning",
        actual_output=(
            "El overfitting ocurre cuando un modelo aprende demasiado bien "
            "los datos de entrenamiento, incluyendo el ruido, y pierde "
            "capacidad de generalizar a datos nuevos."
        ),
        expected_output=(
            "El overfitting es cuando el modelo memoriza el entrenamiento "
            "y falla con datos nuevos."
        )
    )
    metrica = GEval(
        name="Calidad_Didactica",
        criteria=(
            "Evalúa si actual_output explica el concepto de forma clara, "
            "correcta y comprensible para un estudiante universitario. "
            "Penaliza si es incompleto o confuso."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=0.6,
        model=juez,
        verbose_mode=True
    )
    assert_test(test_case, [metrica]) 
