import opik
from opik import track
from openai import OpenAI

# Configurar cliente Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

@track
def generar_examen(tema: str, nivel: str = "universitario", num_preguntas: int = 3):
    prompt = f"""Eres un profesor experto. Genera {num_preguntas} preguntas de examen sobre "{tema}" 
    para nivel {nivel}. Para cada pregunta incluye:
    1. La pregunta
    2. La respuesta correcta
    3. La rúbrica de evaluación (2-3 líneas)
    
    Responde en español."""
    
    response = client.chat.completions.create(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# 3 consultas de prueba
if __name__ == "__main__":
    pruebas = [
        ("Integrales definidas", "universitario"),
        ("Circuitos eléctricos en serie", "universitario"),
        ("Programación orientada a objetos", "universitario")
    ]
    
    for tema, nivel in pruebas:
        print(f"\n📚 Generando examen sobre: {tema}")
        resultado = generar_examen(tema=tema, nivel=nivel)
        print(resultado)
        print("-" * 50) 
