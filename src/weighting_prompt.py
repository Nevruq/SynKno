# File to process a prompt and return a weighing of the most import components of it. Goal: Destilling the Question and remove irrelevant Parts
import call_llm as cllm
import json


def distill_query(user_prompt: str, raw_answer: str) -> dict[str, any]:
    """
    Nimmt eine lange/natürliche Frage und extrahiert:
      - core_query: destillierte Kernfrage
      - subqueries: Liste von { "q": ..., "weight": float }
    """
    print(raw_answer)
    if type(raw_answer) == dict:
        return raw_answer
    # defensive: versuchen, JSON zu parsen
    try:
        data = json.loads(raw_answer)
        print("Json erkannt im try Block")
    except TypeError:
        # Fallback: einfache Struktur bauen
        data = {
            "core_query": user_prompt,
            "subqueries": [{"q": user_prompt, "weight": 1.0}]
        }
    except json.decoder.JSONDecodeError:
        data = {
            "core_query": user_prompt,
            "subqueries": [{"q": user_prompt, "weight": 1.0}]
        }
    """
    # Gewichte normalisieren (optional)
    weights = [sq.get("weight", 1.0) for sq in data["subqueries"]]
    max_w = max(weights) if weights else 1.0
    if max_w > 0:
        for sq in data["subqueries"]:
            sq["weight"] = float(sq.get("weight", 1.0)) / max_w
    """

    return data
