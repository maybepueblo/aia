# vision/graph/router.py
# ─────────────────────────────────────────────────────────────
# Construcción y compilación del grafo LangGraph.
#
# Topología:
#   START → nodo_clasificar → [rutear_por_tipo]
#             ├── BALDA_PRECIOS  → nodo_pipeline_precios → nodo_agregar → END
#             ├── ETIQUETA_NUTRI → nodo_pipeline_nutri   → nodo_agregar → END
#             └── DESCONOCIDO    → nodo_error             → END
# ─────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END

from vision.domain.schemas import VisionState
from vision.graph.nodes import (
    nodo_clasificar,
    nodo_pipeline_precios,
    nodo_pipeline_nutri,
    nodo_agregar,
    nodo_error,
    rutear_por_tipo,
)


def construir_grafo() -> StateGraph:
    g = StateGraph(VisionState)

    g.add_node("nodo_clasificar",       nodo_clasificar)
    g.add_node("nodo_pipeline_precios", nodo_pipeline_precios)
    g.add_node("nodo_pipeline_nutri",   nodo_pipeline_nutri)
    g.add_node("nodo_agregar",          nodo_agregar)
    g.add_node("nodo_error",            nodo_error)

    g.set_entry_point("nodo_clasificar")
    g.add_conditional_edges(
        "nodo_clasificar",
        rutear_por_tipo,
        {
            "nodo_pipeline_precios": "nodo_pipeline_precios",
            "nodo_pipeline_nutri":   "nodo_pipeline_nutri",
            "nodo_error":            "nodo_error",
        },
    )
    g.add_edge("nodo_pipeline_precios", "nodo_agregar")
    g.add_edge("nodo_pipeline_nutri",   "nodo_agregar")
    g.add_edge("nodo_agregar",          END)
    g.add_edge("nodo_error",            END)
    return g


# Singleton compilado — se crea una vez al importar el módulo
grafo_vision = construir_grafo().compile()
print("✓ Grafo LangGraph compilado")
