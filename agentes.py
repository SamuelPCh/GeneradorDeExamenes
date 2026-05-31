from crewai import Agent, Task, Crew

# Agente 1 — Generador de preguntas
agente_generador = Agent(
    role="Generador de Exámenes",
    goal="Generar preguntas de selección múltiple teóricas y conceptuales sobre un tema académico",
    backstory="""Eres un profesor universitario experto en crear evaluaciones académicas.
    Tu especialidad es formular preguntas claras que evalúen comprensión conceptual.""",
    llm="ollama/qwen2.5:3b",
    verbose=True
)

# Agente 2 — Revisor pedagógico
agente_revisor = Agent(
    role="Revisor Pedagógico",
    goal="Revisar y mejorar la calidad pedagógica de las preguntas generadas",
    backstory="""Eres un experto en pedagogía universitaria con amplia experiencia
    evaluando la calidad de exámenes. Verificas que las preguntas sean claras,
    relevantes y bien formuladas.""",
    llm="ollama/qwen2.5:3b",
    verbose=True
)

def ejecutar_agentes(tema, nivel, num_preguntas):
    tarea_generacion = Task(
        description=f"""Genera exactamente {num_preguntas} preguntas teóricas y conceptuales
        de selección múltiple de nivel {nivel} sobre el tema: {tema}.
        NO uses LaTeX ni símbolos matemáticos. Solo texto plano.
        
        Usa EXACTAMENTE este formato:
        
        PUNTO 1
        PREGUNTA: enunciado conceptual
        A: opción a
        B: opción b
        C: opción c
        D: opción d
        CORRECTA: letra correcta
        RUBRICA: criterio de evaluación
        
        Repite para todos los {num_preguntas} puntos.""",
        agent=agente_generador,
        expected_output=f"{num_preguntas} preguntas en formato PUNTO/PREGUNTA/A/B/C/D/CORRECTA/RUBRICA"
    )

    tarea_revision = Task(
        description=f"""Revisa las preguntas generadas sobre {tema} y verifica que:
        1. Sean claras y comprensibles
        2. Evalúen conceptos teóricos sin fórmulas ni cálculos
        3. Las opciones sean plausibles y bien redactadas
        4. Las rúbricas sean útiles para el profesor
        
        Devuelve TODAS las preguntas corregidas en el mismo formato:
        
        PUNTO 1
        PREGUNTA: enunciado
        A: opción a
        B: opción b
        C: opción c
        D: opción d
        CORRECTA: letra
        RUBRICA: criterio""",
        agent=agente_revisor,
        expected_output=f"{num_preguntas} preguntas revisadas en formato PUNTO/PREGUNTA/A/B/C/D/CORRECTA/RUBRICA"
    )

    crew = Crew(
        agents=[agente_generador, agente_revisor],
        tasks=[tarea_generacion, tarea_revision],
        verbose=True
    )

    resultado = crew.kickoff()
    return str(resultado)