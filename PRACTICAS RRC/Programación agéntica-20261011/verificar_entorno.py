#!/usr/bin/env python
"""
Verificación completa del entorno del seminario.

Ejecútalo diez minutos antes de clase:

    .venv/bin/python verificar_entorno.py

Comprueba, una por una, todas las capacidades que usan las libretas. Si algo
falla, imprime el motivo real y qué hacer. Sale con código 1 si hay algún fallo,
de modo que se puede encadenar en un script de arranque.

Por qué existe: un modelo puede aparecer en `ollama list` y estar inservible.
Comprobarlo en el minuto 5 es infinitamente más barato que en el minuto 150.
"""

from __future__ import annotations

import sys
import traceback
import warnings

warnings.filterwarnings("ignore")  # los avisos de deprecación no son fallos aquí

RESULTADOS: list[tuple[bool, str, str]] = []


def comprobar(nombre: str, funcion, remedio: str = "") -> None:
    """Ejecuta una comprobación y apunta el resultado, sin abortar el resto."""
    try:
        detalle = funcion() or "correcto"
        RESULTADOS.append((True, nombre, str(detalle)[:58]))
    except Exception as exc:  # noqa: BLE001 - queremos capturarlo todo
        RESULTADOS.append((False, nombre, f"{type(exc).__name__}: {exc}"[:58]))
        if remedio:
            RESULTADOS.append((False, "  -> remedio", remedio[:58]))
        if "--traza" in sys.argv:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Comprobaciones
# ---------------------------------------------------------------------------
def revisar_daemon() -> str:
    import comun

    modelos = comun.modelos_listados()
    if not modelos:
        raise RuntimeError(f"sin respuesta en {comun.OLLAMA}")
    return f"{len(modelos)} modelos en {comun.OLLAMA}"


def revisar_modelo() -> str:
    import comun

    modelo = comun.elegir_modelo(verbose=False)
    nube = "nube" if comun.es_nube(modelo) else "local"
    return f"{modelo} ({nube})"


def revisar_tool_calling() -> str:
    from langchain.tools import tool

    import comun

    @tool
    def sonda(valor: str) -> str:
        """Devuelve el valor tal cual. Úsala siempre que se te pida un valor."""
        return valor

    respuesta = comun.llm().bind_tools([sonda]).invoke(
        "Usa la herramienta sonda con el valor 'ok'."
    )
    if not respuesta.tool_calls:
        raise RuntimeError("el modelo no pidió ninguna herramienta")
    return f"tool_calls: {respuesta.tool_calls[0]['name']}"


def revisar_salida_estructurada() -> str:
    from pydantic import BaseModel, Field

    import comun

    class Receta(BaseModel):
        titulo: str = Field(description="Nombre del plato")
        minutos: int = Field(description="Minutos de preparación")

    # method="function_calling" es el único fiable con Ollama: la nube no
    # soporta json_schema y json_mode falla.
    resultado = (
        comun.llm()
        .with_structured_output(Receta, method="function_calling")
        .invoke("Inventa una receta rápida con pollo.")
    )
    return f"{type(resultado).__name__}: {resultado.titulo[:30]}"


def revisar_memoria() -> str:
    from langchain_core.messages import SystemMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    import comun

    modelo = comun.llm()

    def nodo(estado: MessagesState):
        msgs = [SystemMessage(content="Eres un asistente conciso.")] + estado["messages"]
        return {"messages": [modelo.invoke(msgs)]}

    c = StateGraph(MessagesState)
    c.add_node("agente", nodo)
    c.add_edge(START, "agente")
    c.add_edge("agente", END)
    grafo = c.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "verificacion-memoria"}}
    grafo.invoke({"messages": [{"role": "user", "content": "Me llamo Luis."}]}, config=cfg)
    grafo.invoke({"messages": [{"role": "user", "content": "¿Cómo me llamo?"}]}, config=cfg)

    guardados = len(grafo.get_state(cfg).values["messages"])
    if guardados < 4:
        raise RuntimeError(f"solo {guardados} mensajes persistidos")
    return f"{guardados} mensajes en el checkpoint"


def revisar_store() -> str:
    from langgraph.store.memory import InMemoryStore

    almacen = InMemoryStore()
    almacen.put(("prueba",), "clave", {"dato": "valor"})
    item = almacen.get(("prueba",), "clave")
    if not item or item.value.get("dato") != "valor":
        raise RuntimeError("no se pudo leer lo escrito")
    return "escritura y lectura por namespace"


def revisar_interrupt() -> str:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import Command, interrupt

    def nodo(estado: MessagesState):
        decision = interrupt({"pregunta": "¿seguimos?"})
        return {"messages": [{"role": "assistant", "content": f"dijo: {decision}"}]}

    c = StateGraph(MessagesState)
    c.add_node("pausa", nodo)
    c.add_edge(START, "pausa")
    c.add_edge("pausa", END)
    grafo = c.compile(checkpointer=InMemorySaver())

    cfg = {"configurable": {"thread_id": "verificacion-interrupt"}}
    parada = grafo.invoke({"messages": [{"role": "user", "content": "hola"}]}, config=cfg)
    if "__interrupt__" not in parada:
        raise RuntimeError("el grafo no se detuvo")
    final = grafo.invoke(Command(resume="sí"), config=cfg)
    if "sí" not in final["messages"][-1].content:
        raise RuntimeError("no se reanudó con la respuesta")
    return "pausa y reanudación con Command(resume=...)"


def revisar_sqlite() -> str:
    import os
    import sqlite3
    import tempfile

    from langgraph.checkpoint.sqlite import SqliteSaver

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "prueba.db")
        conexion = sqlite3.connect(ruta, check_same_thread=False)
        SqliteSaver(conexion)
        conexion.close()
    return "langgraph-checkpoint-sqlite disponible"


def revisar_versiones() -> str:
    import importlib.metadata as md

    minimas = {"langgraph": (1, 2), "langchain": (1, 4), "langchain-core": (1, 6)}
    problemas = []
    for paquete, minima in minimas.items():
        try:
            partes = md.version(paquete).split(".")
            actual = tuple(int(p) for p in partes[:2])
        except Exception:
            problemas.append(f"{paquete} ausente")
            continue
        if actual < minima:
            problemas.append(f"{paquete} {md.version(paquete)} < {minima}")
    if problemas:
        raise RuntimeError("; ".join(problemas))
    return f"langgraph {md.version('langgraph')} · langchain {md.version('langchain')}"


COMPROBACIONES = [
    ("Daemon de Ollama accesible", revisar_daemon,
     "Arranca el servicio con: ollama serve"),
    ("Versiones de la pila", revisar_versiones,
     "Reinstala con: uv pip install -r requirements.txt"),
    ("Modelo elegido y respondiendo", revisar_modelo,
     "Descarga uno local: ollama pull gemma4:e4b"),
    ("Tool calling (CRÍTICO)", revisar_tool_calling,
     "Usa un modelo con capacidad tools: ollama show <modelo>"),
    ("Salida estructurada", revisar_salida_estructurada,
     "Debe usarse method='function_calling'"),
    ("Memoria de corto plazo", revisar_memoria,
     "Revisa langgraph-checkpoint"),
    ("Memoria de largo plazo (Store)", revisar_store,
     "Revisa langgraph.store.memory"),
    ("Interrupción humana", revisar_interrupt,
     "El checkpointer es obligatorio para interrupt()"),
    ("Persistencia SQLite", revisar_sqlite,
     "uv pip install langgraph-checkpoint-sqlite"),
]


def main() -> int:
    print("=" * 78)
    print("VERIFICACIÓN DEL ENTORNO · Seminario de agentes con LangGraph")
    print("=" * 78)
    print()

    for nombre, funcion, remedio in COMPROBACIONES:
        print(f"  comprobando: {nombre} ...", flush=True)
        comprobar(nombre, funcion, remedio)

    print()
    print("=" * 78)
    print("RESULTADO")
    print("=" * 78)

    ancho = max(len(n) for _, n, _ in RESULTADOS) + 2
    for ok, nombre, detalle in RESULTADOS:
        marca = "OK   " if ok else "FALLO"
        print(f"  [{marca}] {nombre:<{ancho}} {detalle}")

    fallos = sum(1 for ok, n, _ in RESULTADOS if not ok and not n.startswith("  ->"))
    print()
    if fallos:
        print(f"  {fallos} comprobación(es) fallida(s).")
        print()
        print("  Plan de choque, en este orden:")
        print("    1. ollama serve                      (en otra terminal)")
        print("    2. ollama pull gemma4:e4b            (local, hace tool calling)")
        print("    3. SEMINARIO_MODELO=gemma4:e4b .venv/bin/python verificar_entorno.py")
        print()
        print("  Detalle de los errores: añade --traza")
        return 1

    print("  Todo correcto: el seminario puede empezar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
