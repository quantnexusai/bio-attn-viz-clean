"""PubMed abstract fetching utilities."""

import requests
import streamlit as st


@st.cache_data
def fetch_pubmed_abstract(query: str, max_results: int = 1) -> str:
    """Fetch abstracts from PubMed via the EuropePMC API.

    Args:
        query: Search query for PubMed.
        max_results: Maximum number of results to fetch.

    Returns:
        Abstract text or error message.
    """
    api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    params = {"query": query, "format": "json", "resultType": "core", "pageSize": max_results}

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "resultList" in data and "result" in data["resultList"]:
            results = data["resultList"]["result"]
            abstracts = [
                result.get("abstractText", "") for result in results if "abstractText" in result
            ]
            return abstracts[0] if abstracts else "No abstract found for this query."
        else:
            return "No results found for this query."
    except Exception as e:
        return f"Error fetching abstracts: {e}"
