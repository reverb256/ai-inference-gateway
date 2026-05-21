"""
Skill Registry Service for AI Inference Gateway.

Provides a lightweight mechanism to manage and query graduated skills
from the Hermes skill system, enabling skill-aware orchestration
between kagent, .sisyphus/, and hermes components.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Metadata for a graduated skill."""
    name: str
    description: str
    version: str
    author: str
    category: str
    tags: List[str]
    related_skills: List[str]
    created_at: float
    updated_at: float
    file_path: str
    usage_count: int = 0
    last_used: Optional[float] = None


class SkillRegistry:
    """Registry for managing graduated skills from Hermes skill system."""
    
    def __init__(self, skills_directory: Optional[str] = None):
        """
        Initialize the skill registry.
        
        Args:
            skills_directory: Path to Hermes skills directory. 
                            Defaults to ~/.hermes/skills/
        """
        if skills_directory is None:
            skills_directory = os.path.expanduser("~/.hermes/skills")
        
        self.skills_directory = Path(skills_directory)
        self._skills_cache: Dict[str, SkillMetadata] = {}
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300.0  # 5 minutes
        
        # Ensure skills directory exists
        self.skills_directory.mkdir(parents=True, exist_ok=True)
        
        # Load skills on initialization
        self._refresh_cache()
    
    def _refresh_cache(self, force: bool = False) -> None:
        """Refresh the skills cache from disk."""
        current_time = time.time()
        if not force and (current_time - self._cache_timestamp) < self._cache_ttl:
            return
        
        logger.debug("Refreshing skill registry cache from %s", self.skills_directory)
        self._skills_cache.clear()
        
        # Walk through skills directory structure
        for skill_path in self.skills_directory.rglob("SKILL.md"):
            try:
                skill_meta = self._parse_skill_file(skill_path)
                if skill_meta:
                    self._skills_cache[skill_meta.name] = skill_meta
            except Exception as e:
                logger.warning("Failed to parse skill file %s: %s", skill_path, e)
        
        self._cache_timestamp = current_time
        logger.info("Loaded %d skills into registry", len(self._skills_cache))
    
    def _parse_skill_file(self, skill_path: Path) -> Optional[SkillMetadata]:
        """Parse a SKILL.md file and extract metadata."""
        try:
            content = skill_path.read_text(encoding='utf-8')
            
            # Extract YAML frontmatter
            if not content.startswith('---'):
                return None
            
            end_marker = content.find('\n---\n', 4)
            if end_marker == -1:
                return None
            
            frontmatter = content[4:end_marker]
            metadata = {}
            
            # Simple YAML parsing for our needed fields
            for line in frontmatter.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle different types
                    if key in ['tags', 'related_skills']:
                        # Parse list format: [item1, item2] or "- item1\n- item2"
                        if value.startswith('[') and value.endswith(']'):
                            # Simple comma-separated list
                            items = [item.strip().strip('"\'') for item in value[1:-1].split(',') if item.strip()]
                        else:
                            # YAML list format
                            items = [item.strip().lstrip('-').strip() for item in value.split('\n') if item.strip()]
                        metadata[key] = items
                    elif key in ['usage_count']:
                        metadata[key] = int(value) if value.isdigit() else 0
                    elif key in ['created_at', 'updated_at', 'last_used']:
                        metadata[key] = float(value) if value.replace('.', '', 1).isdigit() else 0.0
                    else:
                        metadata[key] = value.strip('"\'')
            
            # Set defaults for required fields
            skill_meta = SkillMetadata(
                name=metadata.get('name', skill_path.parent.name),
                description=metadata.get('description', ''),
                version=metadata.get('version', '1.0.0'),
                author=metadata.get('author', 'Unknown'),
                category=metadata.get('category', 'uncategorized'),
                tags=metadata.get('tags', []),
                related_skills=metadata.get('related_skills', []),
                created_at=metadata.get('created_at', time.time()),
                updated_at=metadata.get('updated_at', time.time()),
                file_path=str(skill_path),
                usage_count=metadata.get('usage_count', 0)
            )
            
            return skill_meta
            
        except Exception as e:
            logger.error("Error parsing skill file %s: %s", skill_path, e)
            return None
    
    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        """Get a specific skill by name."""
        self._refresh_cache()
        return self._skills_cache.get(name)
    
    def list_skills(self, 
                   category: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   author: Optional[str] = None) -> List[SkillMetadata]:
        """
        List skills with optional filtering.
        
        Args:
            category: Filter by category
            tags: Filter by tags (must match all tags)
            author: Filter by author
            
        Returns:
            List of matching SkillMetadata objects
        """
        self._refresh_cache()
        
        skills = list(self._skills_cache.values())
        
        if category:
            skills = [s for s in skills if s.category == category]
        
        if tags:
            skills = [s for s in skills if all(tag in s.tags for tag in tags)]
        
        if author:
            skills = [s for s in skills if s.author == author]
        
        return skills
    
    def search_skills(self, query: str) -> List[SkillMetadata]:
        """
        Search skills by name, description, or tags.
        
        Args:
            query: Search string
            
        Returns:
            List of matching SkillMetadata objects
        """
        self._refresh_cache()
        query_lower = query.lower()
        
        matches = []
        for skill in self._skills_cache.values():
            if (query_lower in skill.name.lower() or 
                query_lower in skill.description.lower() or
                any(query_lower in tag.lower() for tag in skill.tags)):
                matches.append(skill)
        
        return matches
    
    def increment_usage(self, skill_name: str) -> None:
        """Increment usage count for a skill."""
        skill = self.get_skill(skill_name)
        if skill:
            skill.usage_count += 1
            skill.last_used = time.time()
            # Note: In a full implementation, we would persist this back to disk
            logger.debug("Incremented usage for skill %s to %d", skill_name, skill.usage_count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        self._refresh_cache()
        
        categories = {}
        authors = {}
        
        for skill in self._skills_cache.values():
            categories[skill.category] = categories.get(skill.category, 0) + 1
            authors[skill.author] = authors.get(skill.author, 0) + 1
        
        return {
            "total_skills": len(self._skills_cache),
            "categories": categories,
            "authors": authors,
            "cache_timestamp": self._cache_timestamp,
            "cache_ttl": self._cache_ttl
        }


# Global registry instance
skill_registry = SkillRegistry()


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    return skill_registry