from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful medical assistant.\n"
            "Use ONLY the information in <CONTEXT>. If the context is insufficient, say so.\n"
            "Never fabricate data. Return concise, accurate answers with citations like [S1], [S2].\n"
            "If asked about a patient, summarize facts from context and avoid speculative clinical advice.\n"
            "This is not medical diagnosis; include a brief caution if the user asks for treatment decisions.",
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "<CONTEXT>\n{context}\n</CONTEXT>\n\n"
            "Citations:\n{citations}\n\n"
            "Instructions:\n"
            "1) If the answer is fully supported by context, answer and cite.\n"
            "2) If partially supported, clearly state limits and cite only supported parts.\n"
            "3) If unsupported, say: 'The provided context does not contain that information.'",
        ),
    ]
)