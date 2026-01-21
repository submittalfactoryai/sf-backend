from fastapi import APIRouter, Request, Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Dict, Any
from decimal import Decimal
from pathlib import Path
import datetime, time, logging, csv, threading

# auth / models
from core.security import get_current_active_user
from models import User
from core.logger import log_action  # used by the helpers below

# Import the same extractor you already use
try:
    from services.extract_pds_add_more_service import extract_pdf_links
except ImportError:
    from ..services.extract_pds_add_more_service import extract_pdf_links  # when running as a package

router = APIRouter(prefix="/api", tags=["PDS Links (Add More)"])
logger = logging.getLogger(__name__)

# ---------- Pricing & usage logging (kept local to avoid circular imports) ----------
MODEL_PRICING = {
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.6},
    "unknown_model": {"input": 0.15, "output": 0.60}
}
GROUNDED_SEARCH_FIXED_COST_PER_REQUEST = 0.035  # $35 per 1000 requests

def get_token_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["unknown_model"])
    input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output"]) / 1_000_000
    return input_cost + output_cost

# Write to the same CSV as the main server
LOG_FILE_PATH = (Path(__file__).resolve().parent.parent / "usage_log.csv").resolve()
CSV_HEADER = [
    "Timestamp", "Session ID", "Filename", "Filesize (Bytes)", "Request Type", "Model Used",
    "Input Tokens", "Output Tokens", "Grounded Searches Performed",
    "Token Cost USD", "Fixed Cost USD", "Total Estimated Cost USD"
]
_csv_lock = threading.Lock()

def log_usage_to_csv(
    timestamp, session_id, filename, filesize, request_type, model_used,
    input_tokens, output_tokens, grounded_searches, token_cost, fixed_cost, total_cost
):
    file_exists = LOG_FILE_PATH.is_file()
    with _csv_lock:
        with open(LOG_FILE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or LOG_FILE_PATH.stat().st_size == 0:
                writer.writerow(CSV_HEADER)
            writer.writerow([
                timestamp.isoformat(),
                session_id if session_id else "N/A",
                filename,
                filesize,
                request_type,
                model_used,
                input_tokens,
                output_tokens,
                grounded_searches,
                f"{token_cost:.8f}",
                f"{fixed_cost:.8f}",
                f"{total_cost:.8f}",
            ])

# Small helpers (mirrors of your api_server helpers, kept here to avoid circular imports)
async def log_api_success(
    request: Request, action: str, entity_type: str, entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None, cost: Optional[float] = None,
    start_time: Optional[float] = None
):
    process_time = int((time.time() - start_time) * 1000) if start_time else None
    cost_decimal = Decimal(str(cost)).quantize(Decimal("0.000001")) if cost is not None else None
    log_action(
        db=request.state.db,
        user_id=getattr(request.state, "user", None) and request.state.user.user_id,
        action_type=action,
        entity_type=entity_type,
        entity_id=entity_id,
        user_metadata=metadata or {},
        cost_estimate=cost_decimal,
        process_time=process_time
    )

async def log_api_failure(
    request: Request, action: str, entity_type: str, entity_id: Optional[str] = None,
    error: Optional[str] = None, start_time: Optional[float] = None
):
    process_time = int((time.time() - start_time) * 1000) if start_time else None
    log_action(
        db=request.state.db,
        user_id=getattr(request.state, "user", None) and request.state.user.user_id,
        action_type=f"{action}_Failed",
        entity_type=entity_type,
        entity_id=entity_id,
        user_metadata={"error": str(error)[:500]} if error else {},
        process_time=process_time
    )

# ---------- Request model (same fields, but refresh is forced in code) ----------
class AddMorePDFLinkExtractionRequest(BaseModel):
    product_name: str
    technical_specifications: dict
    manufacturers: dict
    reference: str = ""
    preferred_state: Optional[str] = None

# ---------- New endpoint ----------
@router.post("/extract-pds-links/add-more")
async def add_more_pds_links_endpoint(
    request: Request,
    body: AddMorePDFLinkExtractionRequest,
    current_user: User = Depends(get_current_active_user),
    x_session_id: Optional[str] = Header(None)
):
    start = time.time()
    logger.info(
        f"Received ADD-MORE PDS link extraction for product: {body.product_name}, "
        f"SessionID: {x_session_id}, refresh=FORCED"
    )
    try:
        product_data = body.dict()
        refresh_requested = True  # <-- always refresh for ADD MORE

        # Same core call as your /api/extract-pds-links
        extraction = await run_in_threadpool(extract_pdf_links, product_data, refresh_requested)
        results = extraction.get("results", [])
        prompt_tokens = extraction.get("prompt_tokens", 0)
        completion_tokens = extraction.get("completion_tokens", 0)
        model_used = extraction.get("model_used", "unknown_model")

        # Cost + CSV logging (same math)
        token_cost = get_token_cost(model_used, prompt_tokens, completion_tokens)
        fixed_cost = GROUNDED_SEARCH_FIXED_COST_PER_REQUEST
        total_cost = token_cost + fixed_cost

        now = datetime.datetime.now(datetime.timezone.utc)
        filename_for_log = f"pds_add_more_for_{product_data.get('product_name', 'unknown_product')}"
        await run_in_threadpool(
            log_usage_to_csv,
            timestamp=now,
            session_id=x_session_id,
            filename=filename_for_log,
            filesize=0,
            request_type="grounded_search_llm_add_more",
            model_used=model_used,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            grounded_searches=1,
            token_cost=token_cost,
            fixed_cost=fixed_cost,
            total_cost=total_cost
        )

        # Audit
        await log_api_success(
            request=request,
            action="SmartSearchAddMore",
            entity_type="Product",
            entity_id=body.product_name,
            metadata={
                "pdf_name": filename_for_log,
                "product_name": body.product_name,
                "refresh": True,
                "results_count": len(results),
                "model_used": model_used,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            },
            cost=total_cost,
            start_time=start
        )

        logger.info(
            f"PDS Links (ADD MORE, REFRESHED) - Product: {product_data.get('product_name','N/A')}, "
            f"SessionID: {x_session_id}, Model: {model_used}, "
            f"Input: {prompt_tokens}, Output: {completion_tokens}, "
            f"TokenCost: ${token_cost:.6f}, FixedCost: ${fixed_cost:.6f}, TotalCost: ${total_cost:.6f}"
        )
        logger.info(f"ADD MORE results count: {len(results)}")
        return {"results": results}

    except Exception as e:
        await log_api_failure(
            request=request,
            action="SmartSearchAddMore",
            entity_type="Product",
            entity_id=getattr(body, "product_name", None),
            error=str(e),
            start_time=start
        )
        logger.error(f"Error during ADD-MORE PDS link extraction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during ADD-MORE PDF link extraction: {str(e)}"
        )
