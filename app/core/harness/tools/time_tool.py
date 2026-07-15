"""Current-time tool — a real external API call, not a wrapped ``datetime.now()``.

Demonstrates the "tool calls an external service" pattern generically (the
same shape as calling an internal company API for order status, inventory
levels, etc.) rather than faking it with a local clock read.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain_core.tools import tool

from app.core.logging import logger

_TIME_API = "https://timeapi.io/api/time/current/zone"


@tool
async def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """Get the current date and time in a given IANA timezone.

    Args:
        timezone: An IANA timezone name, e.g. "Asia/Shanghai", "UTC",
            "America/New_York". Defaults to Shanghai time.

    Returns:
        A string with the current date and time, noting whether it came from
        the external time API or a local fallback.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_TIME_API, params={"timeZone": timezone})
            response.raise_for_status()
            data = response.json()
            return f"{data['dateTime']} ({timezone}, via timeapi.io)"
    except Exception as e:
        logger.warning("time_api_call_failed_using_local_fallback", timezone=timezone, error=str(e))
        try:
            now = datetime.now(ZoneInfo(timezone))
        except Exception:
            now = datetime.now(ZoneInfo("UTC"))
            timezone = "UTC"
        return f"{now.isoformat()} ({timezone}, local fallback — time API unreachable)"
