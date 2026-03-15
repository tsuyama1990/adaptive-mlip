from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.core.preflight import PreflightManager
from pyacemaker.domain_models.compiler import SemanticCompiler
from pyacemaker.domain_models.scenario import IntentRequest

router = APIRouter(prefix="/api/v1/intent", tags=["intent"])


@router.post("/compile")
async def compile_intent(intent: IntentRequest) -> Any:
    """
    Compiles a high-level visual IntentRequest into a full PyAceConfig payload.
    Performs Preflight checks prior to returning.
    """
    try:
        config = SemanticCompiler.compile(intent)
    except CompilerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        manager = PreflightManager()
        report = manager.run(config)

        if report.errors:
            # We must return the report in the response body.
            # We'll return a 400 JSONResponse with the diagnostic report dumped.
            return JSONResponse(
                status_code=400,
                content=report.model_dump()
            )

        # If no errors, return HTTP 200 with the config.
        # Since the Spec mentions "return an HTTP 200 OK along with the report",
        # but also "return the compiled configuration", we can just return a dict
        # or since the endpoint type hint was PyAceConfig previously, let's keep it as is,
        # but with Python FastAPI we can return a dict containing both or just the config
        # according to the tests. Existing tests expect config to be returned.
        # However, UAT-06 explicitly checks `errors` array in successful preflights?
        # Actually UAT 06-A says: "The test suite must rigorously assert that the returned DiagnosticReport JSON object contains empty arrays for the errors and warnings keys, and that the info array contains a single message stating "Preflight validation completed successfully."
        # If the test expects the report in the response, we should return both.
        # But wait! Integration tests in `test_api_endpoints.py` assert `response_data["project_name"] == DEFAULT_PROJECT_NAME` meaning the direct config fields are accessed at root.
        # So we can merge the report into the config dictionary, or just return the config dictionary.
        # But the config is a strict Pydantic model (`extra="forbid"`), so it cannot include `errors`.
        # How to satisfy both? Let's return the JSON dumped dict and append report to it manually, it bypasses Pydantic model response serialization because we return a dict.

        resp_dict = config.model_dump()
        resp_dict.update(report.model_dump())
        return JSONResponse(status_code=200, content=resp_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preflight failed: {e}") from e
