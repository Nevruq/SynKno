def rag_answer(user_prompt: str, vector_db: VectorDB, llm_context_model=None) -> str:
    """
    Komplettes Beispiel:
    1. Query destillieren
    2. Gewichtete Multi-Query-Retrieval
    3. LLM mit Kontext füttern und Antwort generieren
    """
    # 1) Distillation
    distilled = distill_query(user_prompt)
    print("Distilled core_query:", distilled["core_query"])
    print("Subqueries:", distilled["subqueries"])

    # 2) Retrieval
    top_docs = retrieve_weighted(distilled, vector_db=vector_db,
                                 k_per_subquery=8, top_n_final=10)

    context_chunks = [d["text"] for d in top_docs]

    system_prompt = (
        "Du bist ein hilfreicher Assistent in einem RAG-System.\n"
        "Nutze NUR die bereitgestellten Dokumente, um zu antworten.\n"
        "Wenn die Antwort nicht sicher aus den Dokumenten hervorgeht, "
        "sage ehrlich, dass die Informationen fehlen.\n"
    )

    context_str = ""
    for i, c in enumerate(context_chunks):
        context_str += f"[Dokument {i+1}]\n{c}\n\n"

    user_for_llm = (
        f"User-Frage:\n{user_prompt}\n\n"
        f"Destillierte Kernfrage:\n{distilled['core_query']}\n\n"
        f"Kontext-Dokumente:\n{context_str}\n"
        "Bitte beantworte die Frage so präzise wie möglich."
    )

    answer = call_llm(system_prompt=system_prompt, user_prompt=user_for_llm)
    return answer
