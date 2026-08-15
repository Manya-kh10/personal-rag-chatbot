import os
import google.generativeai as genai
from rag_pipeline import retrieve_context

def run_agent(user_message: str, provider: str = "Google Gemini (Cloud)", api_key: str = None) -> dict:
    """
    Executes an agent turn using Gemini or Groq with tool-use capability.
    
    Parameters:
        user_message (str): The user's input query.
        provider (str): LLM provider ("Google Gemini (Cloud)" or "Groq (Cloud)")
        api_key (str, optional): API Key. Defaults to environment variable (GOOGLE_API_KEY or GROQ_API_KEY).
        
    Returns:
        dict: {"answer": str, "used_rag": bool}
    """
    if "Gemini" in provider:
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API Key is required for Gemini Agent. Please set GOOGLE_API_KEY environment variable or enter it in the sidebar.")
            
        genai.configure(api_key=key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        
        # Define tool schema for retrieve_context
        retrieve_tool = genai.protos.FunctionDeclaration(
            name="retrieve_context",
            description="Retrieves relevant document chunks from the uploaded knowledge base using hybrid search and cross-encoder reranking. Call this whenever the question requires specific factual information from uploaded documents.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The target search query to retrieve document context for."
                    }
                },
                "required": ["query"]
            }
        )
        
        tool = genai.protos.Tool(function_declarations=[retrieve_tool])
        model = genai.GenerativeModel(model_name=model_name, tools=[tool])
        
        chat = model.start_chat()
        response = chat.send_message(user_message)
        
        used_rag = False
        
        if response.parts:
            for part in response.parts:
                if fn := part.function_call:
                    if fn.name == "retrieve_context":
                        used_rag = True
                        query = fn.args.get("query", user_message) if hasattr(fn.args, "get") else fn.args["query"]
                        result = retrieve_context(query)
                        
                        # Return function execution output back to Gemini
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name="retrieve_context",
                                        response={"result": result}
                                    )
                                )]
                            )
                        )
                        break
                        
        answer_text = response.text if hasattr(response, "text") else str(response)
        return {"answer": answer_text, "used_rag": used_rag}

    elif "Groq" in provider:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key is required for Groq Agent. Please set GROQ_API_KEY environment variable or enter it in the sidebar.")
            
        from langchain_groq import ChatGroq
        from langchain_core.tools import tool
        
        @tool
        def retrieve_context_tool(query: str) -> str:
            """Retrieves relevant document chunks from the uploaded knowledge base using hybrid search and cross-encoder reranking. Call this whenever the question requires specific factual information from uploaded documents."""
            return retrieve_context(query)
            
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=key)
        llm_with_tools = llm.bind_tools([retrieve_context_tool])
        
        messages = [
            ("system", "You are a helpful personal assistant. You ONLY have access to 'retrieve_context_tool'. Do NOT attempt to call any other function or unprovided tool. If the user query is a general question, math problem, or greeting, answer directly without invoking tools."),
            ("user", user_message)
        ]
        
        try:
            res = llm_with_tools.invoke(messages)
        except Exception:
            res = llm.invoke(user_message)
            
        used_rag = False
        
        if hasattr(res, "tool_calls") and res.tool_calls:
            used_rag = True
            tool_call = res.tool_calls[0]
            query = tool_call.get("args", {}).get("query", user_message)
            context_result = retrieve_context(query)
            
            prompt = f"""You are a helpful personal assistant. Use the following context retrieved from the document corpus to answer the question.
If the answer isn't in the context, state that you don't know.

Context:
{context_result}

User Question: {user_message}
Answer:"""
            final_res = llm.invoke(prompt)
            answer_text = final_res.content
        else:
            answer_text = res.content
            
        return {"answer": answer_text, "used_rag": used_rag}

        
    else:
        raise ValueError(f"Provider '{provider}' is not supported for Agentic mode. Please select Google Gemini or Groq.")
