"""
comun.py — Utilidades compartidas del seminario «Agentes con LangGraph».

Una sola fuente de verdad para:
  * elegir el modelo (nube gratuita de Ollama, con retroceso a local),
  * construir el LLM apuntando al daemon local,
  * inspeccionar el grafo y los mensajes.

Todos los notebooks importan de aquí. Si algo del entorno cambia,
se arregla en un solo sitio y los cinco notebooks siguen funcionando.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# 1. Dónde está Ollama
# --------------------------------------------------------------------------
# El daemon local expone tanto la API nativa (/api/...) como la compatible
# con OpenAI (/v1/...). Si ya has hecho `ollama signin`, los modelos de nube
# funcionan a través de este mismo daemon: no necesitas OLLAMA_API_KEY.
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
BASE_URL_OPENAI = f"{OLLAMA}/v1"

# --------------------------------------------------------------------------
# 2. Qué modelo usar
# --------------------------------------------------------------------------
# Candidatos en orden de preferencia. El primero que CONTESTE de verdad gana.
#   * nube gratuita  -> gemma4:31b-cloud (verificado: tool calling en paralelo)
#   * local offline  -> gemma4:e4b       (verificado: tool calling)
# Un modelo puede aparecer en `ollama list` y estar inservible: retirado,
# fuera de la cuota gratuita, o sin descargar. Por eso aquí no basta con
# consultar la lista: hay que enviarle un mensaje de verdad.
CANDIDATOS = [
    os.getenv("SEMINARIO_MODELO"),  # 0) lo que diga el alumno manda
    "gemma4:31b-cloud",             # 1) nube gratuita (verificado 2026-10)
    "gpt-oss:20b-cloud",            # 2) nube gratuita (alternativa)
    "gemma4:e4b",                   # 3) local, sin red
]

_modelo_activo: str | None = None


def _peticion(ruta: str, payload: dict | None = None, timeout: int = 30) -> dict:
    """GET/POST contra la API nativa de Ollama."""
    url = f"{OLLAMA}{ruta}"
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def modelos_listados() -> list[str]:
    """Los modelos que `ollama list` dice que tienes. Puede mentir."""
    try:
        return [m["name"] for m in _peticion("/api/tags")["models"]]
    except Exception as exc:  # daemon apagado, puerto ocupado, ...
        print(f"[preflight] No puedo hablar con Ollama en {OLLAMA}: {exc}")
        return []


def probar_modelo(modelo: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Envía un mensaje mínimo y comprueba que el modelo responde.

    Devuelve (ok, detalle). El detalle es el motivo del fallo, tal cual lo
    da Ollama: es la información más útil que verá el alumno.
    """
    try:
        data = _peticion(
            "/api/chat",
            {
                "model": modelo,
                "stream": False,
                "messages": [{"role": "user", "content": "ping"}],
                # num_predict=1 para no gastar cuota ni tiempo: solo queremos
                # saber si el modelo está vivo.
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if "error" in data:
        # Aquí caen los casos reales: modelo retirado, cuota gratuita agotada...
        return False, data["error"]
    return True, "responde correctamente"


def elegir_modelo(forzar: str | None = None, verbose: bool = True) -> str:
    """
    Devuelve el primer modelo que responde de verdad.

    Se cachea en el módulo: los notebooks lo llaman muchas veces y no
    queremos sondear la nube en cada celda.
    """
    global _modelo_activo
    if forzar:
        _modelo_activo = forzar
        return forzar
    if _modelo_activo:
        return _modelo_activo

    visto = set()
    for candidato in CANDIDATOS:
        if not candidato or candidato in visto:
            continue
        visto.add(candidato)
        ok, detalle = probar_modelo(candidato)
        if verbose:
            marca = "OK  " if ok else "FALLO"
            print(f"[preflight] {marca} {candidato:24} {detalle[:110]}")
        if ok:
            _modelo_activo = candidato
            if verbose:
                print(f"[preflight] Modelo activo: {candidato}")
            return candidato

    raise RuntimeError(
        "Ningún modelo disponible. Comprueba que el daemon está encendido "
        f"({OLLAMA}) y que tienes al menos un modelo: `ollama pull gemma4:e4b`"
    )


def es_nube(modelo: str | None = None) -> bool:
    """¿El modelo activo vive en la nube o en disco?"""
    return "-cloud" in (modelo or elegir_modelo(verbose=False))


# --------------------------------------------------------------------------
# 3. El LLM
# --------------------------------------------------------------------------
def llm(temperatura: float = 0.0, modelo: str | None = None, **extra):
    """
    ChatOpenAI apuntando al daemon local.

    La clave 'ollama' es obligatoria pero Ollama la ignora: es un detalle
    que confunde mucho al principio. Los modelos de nube se autentican con
    la sesión del daemon (`ollama signin`), no con esta clave.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=modelo or elegir_modelo(verbose=False),
        base_url=BASE_URL_OPENAI,
        api_key="ollama",
        temperature=temperatura,
        **extra,
    )


# --------------------------------------------------------------------------
# 4. Ayudas para clase
# --------------------------------------------------------------------------
def _slug(texto: str) -> str:
    """Convierte un título en un nombre de fichero seguro."""
    import re
    import unicodedata

    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    limpio = re.sub(r"[^a-zA-Z0-9]+", "-", sin_tildes).strip("-").lower()
    return limpio or "grafo"


def mermaid_legible(codigo: str) -> str:
    """
    Retoca el Mermaid que genera LangGraph para que sea legible en clase.

    LangGraph rotula la entrada y la salida con HTML y espacios duros
    (`__start__([<p>__start__</p>])`, `-. &nbsp;tools&nbsp; .->`), que se ven mal
    en el diagrama. Las sustituciones están verificadas con mermaid-cli: el
    resultado sigue siendo Mermaid válido.
    """
    return (
        codigo.replace("[<p>__start__</p>]", "[inicio]")
        .replace("[<p>__end__</p>]", "[fin]")
        .replace("&nbsp;", " ")
        # `-. &nbsp;tools&nbsp; .->` deja dos espacios al sustituir; la
        # indentacion usa tabuladores, asi que colapsar espacios es inocuo.
        .replace("  ", " ")
    )


def _guardar_mermaid(codigo: str, titulo: str, carpeta: str = "grafos"):
    """Deja el diagrama en un fichero .mmd que se puede pegar tal cual."""
    import pathlib

    destino = pathlib.Path(carpeta)
    destino.mkdir(exist_ok=True)
    ruta = destino / f"{_slug(titulo)}.mmd"
    ruta.write_text(codigo + "\n", encoding="utf-8")
    return ruta


def _png_de_mermaid(codigo: str, timeout: int = 15):
    """
    Devuelve el PNG del diagrama usando el servicio público mermaid.ink, o None
    si no hay conexión. Es solo una comodidad: si falla, el código impreso sigue
    siendo válido y se puede pegar en https://mermaid.live

    Nota: el servicio responde 403 al User-Agent por defecto de urllib, así que
    hay que identificarse como un navegador. Sin esa cabecera no devuelve imagen.
    """
    import base64

    try:
        datos = base64.urlsafe_b64encode(codigo.encode()).decode()
        peticion = urllib.request.Request(
            f"https://mermaid.ink/img/{datos}?type=png",
            headers={"User-Agent": "Mozilla/5.0 (compatible; seminario-langgraph)"},
        )
        with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
            contenido = respuesta.read()
        return contenido if contenido[:4] == b"\x89PNG" else None
    except Exception:
        return None


def ver_grafo(app, titulo: str = "Estructura del grafo", dibujar: bool = True) -> None:
    """
    Muestra la estructura del grafo del agente. Es LA herramienta didáctica del
    seminario: los cuatro conceptos se ven aquí dibujados.

    Hace tres cosas, y **el orden importa**:

      1. Dibuja el diagrama aquí mismo, si hay conexión.
      2. Imprime el código Mermaid, para poder copiarlo.
      3. Deja el aviso y la ruta del fichero **al final**.

    El aviso va al final a propósito: así, al seleccionar la salida desde el
    principio, se copia **solo** el diagrama. La versión anterior imprimía una
    cabecera delante y Mermaid la interpretaba como un front-matter YAML mal
    formado, rechazando el diagrama con:

        Diagrams beginning with --- are not valid.

    Por el mismo motivo, ninguna línea de aviso empieza por `---`.

    El diagrama se guarda además en `grafos/<titulo>.mmd`, un fichero limpio y
    directamente utilizable.
    """
    codigo = mermaid_legible(app.get_graph().draw_mermaid())
    ruta = _guardar_mermaid(codigo, titulo)

    dibujado = False
    if dibujar:
        png = _png_de_mermaid(codigo)
        if png:
            try:
                from IPython.display import Image, display

                display(Image(data=png))
                dibujado = True
            except ImportError:
                pass

    print(codigo)
    print()
    print(f"[grafo] {titulo}  ·  guardado en {ruta}")
    if not dibujado:
        print("[grafo] pega el bloque de arriba en https://mermaid.live")


def ver_mensajes(resultado: dict | list, max_chars: int = 220) -> None:
    """
    Recorre los mensajes del estado y muestra quién dijo qué.

    Enseña la mecánica real del bucle: human -> ai(tool_calls) -> tool -> ai.
    Ver los tool_calls es la única forma de saber si el modelo usó las
    herramientas o se lo inventó todo.
    """
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    for i, m in enumerate(mensajes):
        tipo = type(m).__name__
        llamadas = getattr(m, "tool_calls", None) or []
        contenido = (m.content or "")
        if not isinstance(contenido, str):
            contenido = json.dumps(contenido, ensure_ascii=False)
        contenido = contenido.replace("\n", " ")[:max_chars]
        etiqueta = tipo
        if llamadas:
            nombres = ", ".join(f"{c['name']}({c.get('args')})" for c in llamadas)
            etiqueta += f"  ->  TOOL_CALLS: {nombres}"
        print(f"[{i:02d}] {etiqueta:28} {contenido}")


def cronometro(texto: str = "bloque"):
    """Context manager para medir cuánto tarda una llamada (la nube es lenta)."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        t0 = time.perf_counter()
        yield
        print(f"[tiempo] {texto}: {time.perf_counter() - t0:.1f} s")

    return _cm()


def seccion(titulo: str) -> None:
    """Separador visual para las libretas."""
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
