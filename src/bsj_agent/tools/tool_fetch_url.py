from typing import Any, Dict

import textwrap


def tool_fetch_url(url: str) -> Dict[str, Any]:
    """
    Stub fetch tool. Replace with real HTTP fetch (e.g., requests) and parsing.
    """
    print(f"[WARN] tool_fetch_url is a stub. Not fetching: {url}")
    return {
        "url": url,
        "status": 200,
        "content": textwrap.dedent(
            f"""
            <html>
              <head><title>Stub fetch for {url}</title></head>
              <body>
                <p>This is placeholder content. Integrate a real fetcher.</p>
              </body>
            </html>
            """
        ).strip(),
    }
