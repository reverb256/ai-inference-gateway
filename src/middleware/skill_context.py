"""
Skill Context Middleware for AI Inference Gateway.

Provides middleware that injects skill context into agent requests,
enabling skill-aware prompting and orchestration between kagent,
.sisyphus/, and hermes components.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from fastapi import Request, HTTPException
from ...services.skill_registry import get_skill_registry, SkillMetadata

logger = logging.getLogger(__name__)


class SkillContextMiddleware:
    """
    Middleware to inject skill context into agent requests.
    
    This middleware:
    1. Extracts skill requirements from incoming requests
    2. Enriches request context with skill-specific guidance
    3. Injects skill constraints into agent prompts
    4. Validates that required skills are available/configured
    """
    
    def __init__(self):
        self.skill_registry = get_skill_registry()
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process an incoming request to enrich with skill context.
        
        Args:
            request: The FastAPI Request object
            context: A dict for passing state to other middleware
            
        Returns:
            Tuple of (should_continue, optional_error):
            - should_continue: False to short-circuit the pipeline
            - optional_error: HTTPException if blocking the request
        """
        if not self.enabled:
            return True, None
        
        try:
            # Only process chat completion requests for now
            if request.method != "POST" or not request.url.path.endswith("/v1/chat/completions"):
                return True, None
            
            # Get the request body
            body = await request.json()
            
            # Extract skill requirements from request
            skill_requirements = self._extract_skill_requirements(body)
            
            if skill_requirements:
                # Enrich context with skill information
                skill_context = await self._build_skill_context(skill_requirements)
                context["skill_context"] = skill_context
                
                # Modify the request to include skill guidance
                modified_body = await self._inject_skill_guidance(body, skill_context)
                
                # Update the request with modified body
                # Note: This requires replacing the request body, which is complex in FastAPI
                # For now, we'll store it in context for downstream use
                context["modified_body"] = modified_body
                
                logger.debug("Injected skill context for requirements: %s", skill_requirements)
            
            return True, None
            
        except Exception as e:
            logger.error("Error in skill context middleware: %s", e)
            # Don't block the request on skill context errors
            return True, None
    
    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process an outgoing response to add skill usage tracking.
        
        Args:
            response: The response dict to modify
            context: State from request processing
            
        Returns:
            Modified response dict
        """
        if not self.enabled:
            return response
        
        try:
            # Track skill usage if skills were used in this request
            skill_context = context.get("skill_context")
            if skill_context and skill_context.get("applied_skills"):
                for skill_name in skill_context["applied_skills"]:
                    self.skill_registry.increment_usage(skill_name)
                
                logger.debug("Tracked usage for skills: %s", 
                           skill_context["applied_skills"])
            
            return response
            
        except Exception as e:
            logger.error("Error tracking skill usage: %s", e)
            return response
    
    def _extract_skill_requirements(self, body: dict) -> List[str]:
        """
        Extract skill requirements from request body.
        
        Looks for:
        1. Explicit skill requirements in metadata
        2. Implicit requirements based on content analysis
        3. References to specific skills in messages
        
        Args:
            body: Request body dictionary
            
        Returns:
            List of required skill names
        """
        requirements = []
        
        # Check for explicit skill requirements
        if "metadata" in body and "skills" in body["metadata"]:
            skills = body["metadata"]["skills"]
            if isinstance(skills, list):
                requirements.extend([s for s in skills if isinstance(s, str)])
            elif isinstance(skills, str):
                requirements.append(skills)
        
        # Check for skill references in messages
        if "messages" in body:
            for message in body["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        # Look for skill reference patterns
                        # e.g., "use skill: skill-name" or "require skill: skill-name"
                        import re
                        skill_patterns = [
                            r'use\s+skill:\s*([a-zA-Z0-9\-_]+)',
                            r'require\s+skill:\s*([a-zA-Z0-9\-_]+)',
                            r'\[skill:\s*([a-zA-Z0-9\-_]+)\]',
                            r'<skill>\s*([a-zA-Z0-9\-_]+)\s*</skill>'
                        ]
                        
                        for pattern in skill_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            requirements.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_requirements = []
        for req in requirements:
            if req not in seen:
                seen.add(req)
                unique_requirements.append(req)
        
        return unique_requirements
    
    async def _build_skill_context(self, skill_requirements: List[str]) -> Dict[str, Any]:
        """
        Build skill context from requirements.
        
        Args.
            skill_requirements: List of required skill names
            
        Returns:
            Dictionary containing skill context information
        """
        context = {
            "required_skills": skill_requirements,
            "available_skills": [],
            "missing_skills": [],
            "skill_guidance": {},
            "applied_skills": []
        }
        
        for skill_name in skill_requirements:
            skill = self.skill_registry.get_skill(skill_name)
            if skill:
                context["available_skills"].append(skill)
                context["skill_guidance"][skill_name] = {
                    "description": skill.description,
                    "version": skill.version,
                    "category": skill.category,
                    "tags": skill.tags,
                    "related_skills": skill.related_skills,
                    "usage_count": skill.usage_count
                }
                context["applied_skills"].append(skill_name)
            else:
                context["missing_skills"].append(skill_name)
                logger.warning("Required skill '%s' not found in registry", skill_name)
        
        return context
    
    async def _inject_skill_guidance(self, body: dict, skill_context: dict) -> dict:
        """
        Inject skill guidance into the request body.
        
        Args:
            body: Original request body
            skill_context: Built skill context
            
        Returns:
            Modified request body with skill guidance
        """
        # Create a deep copy to avoid modifying the original
        import copy
        modified_body = copy.deepcopy(body)
        
        # Add skill context to metadata
        if "metadata" not in modified_body:
            modified_body["metadata"] = {}
        
        modified_body["metadata"]["skill_context"] = skill_context
        
        # If we have skill guidance, consider adding it to the system prompt
        # or as additional context in the messages
        if skill_context.get("skill_guidance"):
            guidance_parts = []
            for skill_name, guidance in skill_context["skill_guidance"].items():
                guidance_parts.append(
                    f"Skill '{skill_name}' (v{guidance['version']}): {guidance['description']}"
                )
                if guidance['tags']:
                    guidance_parts.append(f"  Tags: {', '.join(guidance['tags'])}")
                if guidance['related_skills']:
                    guidance_parts.append(f"  Related: {', '.join(guidance['related_skills'])}")
            
            if guidance_parts:
                guidance_text = "\n\nAvailable Skills Guidance:\n" + "\n".join(guidance_parts)
                
                # Try to inject into system message or first user message
                if "messages" in modified_body:
                    # Look for system message first
                    for i, message in enumerate(modified_body["messages"]):
                        if isinstance(message, dict) and message.get("role") == "system":
                            if "content" in message:
                                modified_body["messages"][i]["content"] += guidance_text
                            break
                    else:
                        # No system message found, prepend to first user message or add as system
                        for i, message in enumerate(modified_body["messages"]):
                            if isinstance(message, dict) and message.get("role") == "user":
                                if "content" in message:
                                    modified_body["messages"][i]["content"] = guidance_text + "\n\n" + modified_body["messages"][i]["content"]
                                break
                        else:
                            # No user message found, insert as system message at start
                            modified_body["messages"].insert(0, {
                                "role": "system",
                                "content": guidance_text
                            })
        
        return modified_body