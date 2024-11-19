from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Any

def find_glossary_terms(query: str, glossary: Dict[str, str]) -> List[tuple[str, str]]:
    """
    Find glossary terms that appear in the query.
    Handles both single-word and multi-word terms.
    
    Args:
        query: The user query string
        glossary: Dictionary of terms and their explanations
    
    Returns:
        List of tuples containing matched terms and their explanations
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Find matching terms
    matches = []
    for term, explanation in glossary.items():
        # Convert term to lowercase for case-insensitive matching
        if term.lower() in query_lower:
            matches.append((term, explanation))
    
    return matches

async def get_rewritten_query_using_glossary(
    query: str,
    glossary: Dict[str, str],
    llm: Any,
    language: str = "german"
) -> str:
    """
    Generate a rewritten query using a glossary, but only including relevant terms.
    
    Args:
        query: Original user query
        glossary: Dictionary of terms and explanations
        llm: Azure OpenAI LLM instance
        language: Query language
    Returns:
        str: Rewritten query
    """
    # Find terms from glossary that appear in the query
    matching_terms = find_glossary_terms(query, glossary)
    
    # If no terms found, create minimal prompt without glossary
    if not matching_terms:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are an expert at reformulating queries to be more effective.
                Your task is to rewrite the following search query while preserving the original intent.
                The query has been asked in the context of the University of Graz.
                Please provide the rewritten query in {language}.
                Original query: '{query}'
                """,
            ),
            ("user", "{query}"),
        ])
    else:
        # Create glossary explanation only for matching terms
        relevant_glossary = "\n".join([f"{term}, Explanation: {explanation}" 
                                     for term, explanation in matching_terms])
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are an expert at reformulating queries to be more effective.
                Your task is to rewrite the following search query using the provided glossary terms and explanations, while preserving the original intent.
                The following terms from your query have specific meanings in the context of the University of Graz:
                {relevant_glossary}
                Please provide the rewritten query in {language}, incorporating the specific meanings of these terms.
                Original query: '{query}'
                """,
            ),
            ("user", "{query}"),
        ])

    chain = prompt | llm | StrOutputParser()
    rewritten_query = await chain.ainvoke({"query": query})
    
    print(f"Rewritten Query String: {rewritten_query}")
    
    return rewritten_query