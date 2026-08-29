"""MCP server: fetch allowlisted Australian official student pages.

Run from the backend directory:
    python -m app.rag.mcp_official_fetch
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.rag.official_fetch import (
    OfficialFetchError,
    fetch_official_page,
    list_official_sources,
)


mcp = FastMCP("official-fetch")


@mcp.tool(name="list_official_sources")
def list_official_sources_tool() -> list[dict[str, str]]:
    """List curated official URLs for visa, OSHC, USYD, and accommodation."""
    return list_official_sources()


@mcp.tool(name="fetch_official_page")
async def fetch_official_page_tool(url: str) -> dict[str, str]:
    """Fetch one HTTPS official page. Rejects URLs outside the allowlist."""
    try:
        page = await fetch_official_page(url)
    except OfficialFetchError as exc:
        return {"error": str(exc), "url": url, "title": "", "text": ""}
    except Exception as exc:
        return {"error": f"Fetch failed: {exc}", "url": url, "title": "", "text": ""}
    return {"url": page.url, "title": page.title, "text": page.text, "error": ""}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
