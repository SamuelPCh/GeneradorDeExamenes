import streamlit as st
from openai import OpenAI
from agentes import ejecutar_agentes
import opik
from opik import track

opik.configure(use_local=False)

st.set_page_config(
    page_title="Generador de Exámenes",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1624; }
.main-header { text-align: center; padding: 2rem 0 1rem; }

.badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,142,247,0.15); border: 1px solid rgba(79,142,247,0.3);
    color: #7eb8ff; font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
    padding: 4px 12px; border-radius: 20px; text-transform: uppercase; margin-bottom: 14px;
}
.main-title {
    font-family: 'Playfair Display', serif; color: #f0f4ff;
    font-size: 36px; font-weight: 700; margin: 0 0 6px; line-height: 1.2;
}
.main-subtitle { color: #8892aa; font-size: 14px; margin: 0 0 2rem; }
.section-label {
    font-size: 11px; font-weight: 500; color: #4f8ef7;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }

.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #111111 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input::placeholder { color: #888888 !important; }

.stRadio > div > label > div > p {
    color: #f0f4ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stRadio > div > label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
.stSlider > div > div > div { background: #4f8ef7 !important; }
label, p { color: #8892aa !important; font-family: 'DM Sans', sans-serif !important; }

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d9e75) !important;
    color: white !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin-top: 8px !important;
}
.stDownloadButton > button {
    background: rgba(79,142,247,0.15) !important;
    color: #7eb8ff !important;
    border: 1px solid rgba(79,142,247,0.3) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
}
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #f0f4ff !important;
    font-family: monospace !important;
    font-size: 13px !important;
}
.stFileUploader > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
.stFileUploader label { color: #f0f4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Cliente Ollama ──────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ── Función que construye el LaTeX ──────────────────────────────
def construir_latex(bloques, institucion, nombre_examen, profesor, tema, nivel):
    encabezado = (
        "\\documentclass{article}\n"
        "\\usepackage[spanish]{babel}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=2.5cm}\n"
        "\n"
        "\\begin{document}\n"
        "\n"
        "\\begin{center}\n"
        "{\\Large\\textbf{" + institucion + "}}\\\\[6pt]\n"
        "{\\large\\textbf{" + nombre_examen + "}}\\\\[4pt]\n"
        "{\\normalsize Profesor: " + profesor + "}\\\\[4pt]\n"
        "{\\normalsize Tema: " + tema + " --- Nivel: " + nivel + "}\\\\[4pt]\n"
        "{\\normalsize Fecha: \\today}\n"
        "\\end{center}\n"
        "\n"
        "\\vspace{8pt}\n"
        "\\noindent\\textbf{Nombre:} \\underline{\\hspace{8cm}} "
        "\\quad \\textbf{Código:} \\underline{\\hspace{3cm}}\n"
        "\\vspace{16pt}\n"
    )

    preguntas = ""
    for i, b in enumerate(bloques):
        n = i + 1
        preguntas += (
            f"\n\\noindent\\textbf{{Punto {n}:}} {b.get('pregunta', '')}\n"
            f"\n\\vspace{{6pt}}\n"
            f"\\noindent a) {b.get('a', 'Opción no generada')} \\\\[6pt]\n"
            f"\\noindent b) {b.get('b', 'Opción no generada')} \\\\[6pt]\n"
            f"\\noindent c) {b.get('c', 'Opción no generada')} \\\\[6pt]\n"
            f"\\noindent d) {b.get('d', 'Opción no generada')}\n"
            f"\n\\vspace{{14pt}}\n"
        )

    guia = (
        "\n\\newpage\n"
        "\n\\begin{center}\n"
        "{\\Large\\textbf{Guía de Calificación}}\\\\[4pt]\n"
        "{\\normalsize Solo para el Profesor --- " + nombre_examen + "}\n"
        "\\end{center}\n"
        "\n\\vspace{12pt}\n"
    )
    for i, b in enumerate(bloques):
        n = i + 1
        correcta = b.get("correcta", "A").upper()
        opcion_correcta = b.get(correcta.lower(), "")
        rubrica = b.get("rubrica", "Evalúa la comprensión del concepto.")
        guia += (
            f"\n\\noindent\\textbf{{Punto {n}:}} "
            f"Respuesta correcta: {correcta}) {opcion_correcta}\n"
            f"\n\\vspace{{6pt}}\n"
            f"\\noindent\\textit{{Rúbrica:}} {rubrica}\n"
            f"\n\\vspace{{12pt}}\n"
        )

    return encabezado + preguntas + guia + "\n\\end{document}\n"


def parsear_respuesta(raw):
    bloques = []
    current = {}
    for linea in raw.splitlines():
        l = linea.strip()
        if not l:
            continue
        lu = l.upper()
        if lu.startswith("PUNTO ") or lu.startswith("**PUNTO "):
            if current.get("pregunta"):
                bloques.append(current)
            current = {}
        elif any(lu.startswith(p) for p in ["PREGUNTA:", "**PREGUNTA:"]):
            current["pregunta"] = l.split(":", 1)[1].strip().replace("**", "")
        elif lu.startswith("A:") or lu.startswith("A.)") or lu.startswith("A)") or lu.startswith("A."):
            current["a"] = l.split(")", 1)[-1].split(":", 1)[-1].strip().replace("**", "").strip(". ")
        elif lu.startswith("B:") or lu.startswith("B.)") or lu.startswith("B)") or lu.startswith("B."):
            current["b"] = l.split(")", 1)[-1].split(":", 1)[-1].strip().replace("**", "").strip(". ")
        elif lu.startswith("C:") or lu.startswith("C.)") or lu.startswith("C)") or lu.startswith("C."):
            current["c"] = l.split(")", 1)[-1].split(":", 1)[-1].strip().replace("**", "").strip(". ")
        elif lu.startswith("D:") or lu.startswith("D.)") or lu.startswith("D)") or lu.startswith("D."):
            current["d"] = l.split(")", 1)[-1].split(":", 1)[-1].strip().replace("**", "").strip(". ")
        elif any(lu.startswith(p) for p in ["CORRECTA:", "RESPUESTA:", "RESPUESTA CORRECTA:"]):
            val = l.split(":", 1)[1].strip().replace("**", "")
            current["correcta"] = val[0].upper() if val else "A"
        elif any(lu.startswith(p) for p in ["RUBRICA:", "RÚBRICA:"]):
            current["rubrica"] = l.split(":", 1)[1].strip().replace("**", "")
    if current.get("pregunta"):
        bloques.append(current)
    return bloques


# ── Interfaz ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">⬤ &nbsp;IA Local · Privado</div>
    <div class="main-title">Generador de Exámenes</div>
    <div class="main-subtitle">Crea evaluaciones académicas con inteligencia artificial</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Información institucional</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    institucion = st.text_input("Institución", placeholder="Ej: Universidad Nacional")
with col2:
    profesor = st.text_input("Nombre del profesor", placeholder="Ej: Dr. Juan Pérez")
nombre_examen = st.text_input("Nombre del examen", placeholder="Ej: Parcial 1 — Cálculo Integral")

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Configuración del examen</div>', unsafe_allow_html=True)

tema = st.text_input("Tema", placeholder="Ej: Factorización, Circuitos en serie...")
nivel = st.radio("Nivel de dificultad", options=["Básico", "Medio", "Alto"], horizontal=True)
num_preguntas = st.slider("Número de preguntas", min_value=1, max_value=10, value=3)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Modelo de IA</div>', unsafe_allow_html=True)

modelo_seleccionado = st.radio(
    "Modelo",
    options=[
        "qwen2.5:3b — Modelo principal",
        "qwen2.5:1.5b — Modelo de comparación"
    ],
    horizontal=True,
    label_visibility="hidden"
)
modelo_id = "qwen2.5:3b" if "3b" in modelo_seleccionado else "qwen2.5:1.5b"

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Material de referencia</div>', unsafe_allow_html=True)

opcion_rag = st.radio(
    "Fuente",
    options=["Sin documento — tema libre", "Cargar mi propio documento"],
    label_visibility="hidden"
)

documento_texto = None
if "Cargar" in opcion_rag:
    archivo = st.file_uploader("Sube tu documento (PDF o TXT)", type=["pdf", "txt"])
    if archivo:
        if archivo.type == "text/plain":
            documento_texto = archivo.read().decode("utf-8")
            st.success(f"✅ Documento cargado: {archivo.name}")
        elif archivo.type == "application/pdf":
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(archivo.read()))
                documento_texto = "".join(p.extract_text() for p in reader.pages)
                st.success(f"✅ PDF cargado: {archivo.name} ({len(reader.pages)} páginas)")
            except Exception as e:
                st.error(f"Error al leer el PDF: {e}")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Botón principal ─────────────────────────────────────────────
if st.button("Generar exámenes en LaTeX →"):
    if not tema:
        st.error("⚠️ Por favor ingresa un tema")
    elif not institucion or not profesor or not nombre_examen:
        st.error("⚠️ Por favor completa la información institucional")
    else:
        with st.spinner("Generando examen con agentes de IA, esto puede tardar..."):

            contexto = (
                f"\nBasándote ÚNICAMENTE en este material:\n\n{documento_texto[:3000]}\n"
                if documento_texto else ""
            )

            @track
            def generar_y_registrar(tema, nivel, num_preguntas, modelo):
                return ejecutar_agentes(tema, nivel, num_preguntas)

            raw = generar_y_registrar(tema, nivel, num_preguntas, modelo_id)
            raw = raw.replace("```", "").strip()

            bloques = parsear_respuesta(raw)

        if not bloques:
            st.error(" El modelo no generó preguntas en el formato esperado. Intenta de nuevo.")
        else:
            resultado = construir_latex(
                bloques, institucion, nombre_examen, profesor, tema, nivel
            )

            st.success(f"¡Examen generado con {len(bloques)} preguntas usando agentes de IA!")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            st.markdown(
                '<div class="section-label" style="margin-top:1rem">'
                'Examen y Guía de Calificación</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03);
            border:1px solid rgba(79,142,247,0.2); border-radius:10px;
            padding:10px 14px; color:#8892aa; font-size:12px; margin-bottom:8px;">
            Copia y pega en
            <a href="https://www.overleaf.com" target="_blank"
            style="color:#7eb8ff;">Overleaf</a>
            para visualizar y descargar como PDF
            </div>
            """, unsafe_allow_html=True)

            st.text_area(
                "Resultado",
                value=resultado,
                height=450,
                label_visibility="hidden"
            )

            st.download_button(
                label="⬇️ Descargar Examen y Guía (.tex)",
                data=resultado,
                file_name=f"examen_{tema.replace(' ', '_')}.tex",
                mime="text/plain"
            )