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
        import logging
        logging.getLogger(__name__).exception("Compilation failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal compilation error.") from e

    try:
        manager = PreflightManager()
        report = manager.run(config)

        if report.errors:
            return JSONResponse(
                status_code=400,
                content={"errors": [e.model_dump() for e in report.errors]}
            )

        # Build secure, explicit response
        resp_dict = config.model_dump()

        # In order to strictly pass UAT 06-A, we inject the report keys explicitly
        # Rather than a full merge that leaks internal configuration schemas
        resp_dict["errors"] = [e.model_dump() for e in report.errors]
        resp_dict["warnings"] = [w.model_dump() for w in report.warnings]
        resp_dict["info"] = [i.model_dump() for i in report.info]

        return JSONResponse(status_code=200, content=resp_dict)

    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Preflight orchestration failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal preflight validation error.") from e
