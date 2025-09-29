from server.model_load import chat_with_model

def build_context_and_ask(question, context_docs, persona: str):
    try:
        if context_docs:
            context = "\n\n".join([f"[{doc['source']}]\n{doc['text']}" for doc in context_docs])
        else:
            context = "(No relevant context found in knowledge base.)"

        prompt = (
            f"{persona}\n"
            f"Use the context below to answer the question when relevant.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "- If the context contains relevant information, use it.\n"
            "- If not, answer from general knowledge.\n"
            "- Be clear, concise, and helpful.\n"
            "- Do not include irrelevant context."
        )

        answer = chat_with_model(prompt)
        return answer.strip() if answer else "⚠️ Model returned no response."
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"
