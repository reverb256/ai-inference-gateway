import logging
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from ai_inference_gateway.router import ModelInfo
from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)

class ScopeEnforcerMiddleware(Middleware):
    """
    Enforces model-level access control based on user roles.
    Matches user roles against the 'required_role' defined in the model registry.
    """
    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        # 1. Only enforce scope for chat/completions endpoints
        if not request.url.path.startswith("/v1/chat/completions"):
            return True, None

        try:
            # 2. Extract user identity from request state (set by JWTAuthMiddleware)
            user = getattr(request.state, "user", None)

            # 3. Get the routed model from context
            # The router has already run and decided which model to use
            route_decision = context.get("route_decision")
            if not route_decision:
                # If routing hasn't happened yet, we can't enforce scope based on the actual model
                # However, the pipeline in main.py calls routing BEFORE process_request
                return True, None

            routed_model = route_decision.model

            # 4. Access the router from gateway state to check requirements
            gateway_state = request.app.state.gateway
            router = gateway_state.router
            if not router:
                return True, None

            # 5. Lookup model requirements
            model_info: ModelInfo = router.models.get(routed_model)
            
            if model_info and model_info.required_role:
                required = model_info.required_role
                
                # Block if no user is authenticated
                if user is None:
                    logger.warning(f"Blocked unauthorized access to restricted model: {routed_model}")
                    return False, HTTPException(
                        status_code=401, 
                        detail=f"Authentication required for model {routed_model}"
                    )
                
                # Block if user doesn't have the required role
                if required not in user.roles:
                    logger.warning(f"User {user.id} blocked from {routed_model}. Missing role: {required}")
                    return False, HTTPException(
                        status_code=403, 
                        detail=f"Insufficient permissions. Required role: {required}"
                    )

        except Exception as e:
            logger.error(f"Scope enforcement error: {e}", exc_info=True)
            # Fail closed: block request on unexpected error in security middleware
            return False, HTTPException(
                status_code=500, 
                detail="Internal security enforcement error"
            )

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        # No response modification needed for scope enforcement
        return response
