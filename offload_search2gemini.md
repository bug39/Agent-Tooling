"""
Exploratory pattern detection for Gemini auto-suggestion.

Detects when Claude is doing expensive exploratory work and suggests
using Gemini to offload token costs.
"""

import time
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TriggerType(Enum):
    """Types of exploratory patterns detected."""
    MULTIPLE_SEARCHES = "multiple_searches"
    EXPLORATORY_READING = "exploratory_reading"
    BROAD_GLOB = "broad_glob"
    NOISY_GREP = "noisy_grep"


@dataclass
class SuggestionTrigger:
    """Represents a detected pattern that suggests using Gemini."""
    type: TriggerType
    confidence: str  # 'high', 'medium', 'low'
    estimated_savings: str  # Human-readable estimate
    suggested_query: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Represents a tool call made by Claude."""
    timestamp: float
    tool_name: str
    args: Dict[str, Any]
    result: Optional[Any] = None


class ExploratoryDetector:
    """
    Detects exploratory patterns in tool usage.

    Tracks recent tool calls and identifies patterns that indicate
    Claude is doing expensive discovery work.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize detector.

        Args:
            config: Configuration dict with thresholds (merged with defaults)
        """
        # Merge custom config with defaults
        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.tool_history: List[ToolCall] = []
        self.recent_searches: List[ToolCall] = []
        self.recent_reads: List[ToolCall] = []

    @staticmethod
    def _default_config() -> Dict:
        """Default configuration thresholds."""
        return {
            'min_grep_calls': 3,
            'min_read_calls': 5,
            'min_glob_results': 15,
            'max_grep_results': 20,
            'time_window_seconds': 60,
        }

    def on_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Optional[Any] = None
    ) -> Optional[SuggestionTrigger]:
        """
        Track a tool call and check for exploratory patterns.

        Args:
            tool_name: Name of the tool called
            args: Arguments passed to the tool
            result: Optional result from the tool

        Returns:
            SuggestionTrigger if a pattern is detected, None otherwise
        """
        call = ToolCall(
            timestamp=time.time(),
            tool_name=tool_name,
            args=args,
            result=result
        )

        self.tool_history.append(call)

        # Track specific tool types
        if tool_name in ['Grep', 'Glob']:
            self.recent_searches.append(call)

        elif tool_name == 'Read':
            self.recent_reads.append(call)

        # Check for patterns
        trigger = self._check_patterns()

        # Clean up old history
        self._cleanup_old_entries()

        return trigger

    def _check_patterns(self) -> Optional[SuggestionTrigger]:
        """
        Check for exploratory patterns in recent tool calls.

        Returns:
            SuggestionTrigger if pattern detected, None otherwise
        """
        # Pattern A: Multiple searches
        trigger = self._check_multiple_searches()
        if trigger:
            return trigger

        # Pattern B: Exploratory reading
        trigger = self._check_exploratory_reading()
        if trigger:
            return trigger

        # Pattern C: Broad glob results
        trigger = self._check_broad_glob()
        if trigger:
            return trigger

        # Pattern D: Noisy grep
        trigger = self._check_noisy_grep()
        if trigger:
            return trigger

        return None

    def _check_multiple_searches(self) -> Optional[SuggestionTrigger]:
        """Check for multiple search calls in short time."""
        recent = self._get_recent_calls(
            self.recent_searches,
            window=self.config['time_window_seconds']
        )

        if len(recent) >= self.config['min_grep_calls']:
            # Extract search patterns
            patterns = [
                call.args.get('pattern', '')
                for call in recent
            ]

            # Infer common theme
            theme = self._infer_theme(patterns)

            return SuggestionTrigger(
                type=TriggerType.MULTIPLE_SEARCHES,
                confidence='high',
                estimated_savings='20-30k tokens',
                suggested_query=f"Where is {theme} implemented? List files, functions, and flow.",
                context={'patterns': patterns, 'call_count': len(recent)}
            )

        return None

    def _check_exploratory_reading(self) -> Optional[SuggestionTrigger]:
        """Check for sequential reads without edits."""
        # Count consecutive reads from end of history
        consecutive_reads = 0
        files_read = []

        for call in reversed(self.tool_history):
            if call.tool_name == 'Read':
                consecutive_reads += 1
                files_read.append(call.args.get('file_path', ''))
            elif call.tool_name in ['Write', 'Edit']:
                # Found an edit, stop counting
                break

        if consecutive_reads >= self.config['min_read_calls']:
            # Infer common directory
            common_dir = "the codebase"
            if files_read and len(files_read) > 1:
                try:
                    # Filter out any None or empty paths
                    valid_paths = [p for p in files_read if p]
                    if valid_paths:
                        common_dir = os.path.commonpath(valid_paths)
                        # Make it more readable
                        if common_dir == '':
                            common_dir = "the codebase"
                except (ValueError, TypeError):
                    common_dir = "the codebase"

            return SuggestionTrigger(
                type=TriggerType.EXPLORATORY_READING,
                confidence='high',
                estimated_savings='30-40k tokens',
                suggested_query=f"Explain the architecture and purpose of {common_dir}",
                context={'files_read': files_read, 'count': consecutive_reads}
            )

        return None

    def _check_broad_glob(self) -> Optional[SuggestionTrigger]:
        """Check for glob with many results."""
        # Look at most recent glob call
        for call in reversed(self.tool_history):
            if call.tool_name == 'Glob' and call.result:
                # Check result size
                result_count = len(call.result) if isinstance(call.result, list) else 0

                if result_count >= self.config['min_glob_results']:
                    pattern = call.args.get('pattern', '')

                    return SuggestionTrigger(
                        type=TriggerType.BROAD_GLOB,
                        confidence='medium',
                        estimated_savings='15-25k tokens',
                        suggested_query=f"What's in {pattern}? Summarize structure and purpose",
                        context={'pattern': pattern, 'result_count': result_count}
                    )
                break

        return None

    def _check_noisy_grep(self) -> Optional[SuggestionTrigger]:
        """Check for grep with many results."""
        # Look at most recent grep call
        for call in reversed(self.tool_history):
            if call.tool_name == 'Grep' and call.result:
                # Count results (simplified - would need actual output parsing)
                result_lines = str(call.result).count('\n')

                if result_lines >= self.config['max_grep_results']:
                    pattern = call.args.get('pattern', '')

                    return SuggestionTrigger(
                        type=TriggerType.NOISY_GREP,
                        confidence='high',
                        estimated_savings='25-35k tokens',
                        suggested_query=f"Find all instances of '{pattern}' and explain usage patterns",
                        context={'pattern': pattern, 'result_count': result_lines}
                    )
                break

        return None

    def _infer_theme(self, patterns: List[str]) -> str:
        """
        Infer common theme from search patterns.

        Args:
            patterns: List of search patterns

        Returns:
            Inferred theme as a string
        """
        # Combine patterns and look for keywords
        combined = ' '.join(patterns).lower()

        # Keyword mappings
        themes = {
            'authentication': ['auth', 'login', 'permission', 'access', 'user'],
            'validation': ['validate', 'check', 'verify', 'sanitize'],
            'error handling': ['error', 'exception', 'catch', 'try'],
            'testing': ['test', 'assert', 'mock', 'fixture'],
            'configuration': ['config', 'setting', 'env', 'option'],
            'database': ['db', 'query', 'sql', 'table', 'model'],
        }

        for theme, keywords in themes.items():
            if any(kw in combined for kw in keywords):
                return theme

        # Fallback: use most common pattern
        return patterns[0] if patterns else 'this functionality'

    def _get_recent_calls(
        self,
        calls: List[ToolCall],
        window: int
    ) -> List[ToolCall]:
        """
        Get calls within time window.

        Args:
            calls: List of tool calls
            window: Time window in seconds

        Returns:
            List of recent calls within window
        """
        cutoff = time.time() - window
        return [call for call in calls if call.timestamp >= cutoff]

    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory bloat."""
        cutoff = time.time() - (self.config['time_window_seconds'] * 2)

        self.tool_history = [
            call for call in self.tool_history
            if call.timestamp >= cutoff
        ]

        self.recent_searches = [
            call for call in self.recent_searches
            if call.timestamp >= cutoff
        ]

        self.recent_reads = [
            call for call in self.recent_reads
            if call.timestamp >= cutoff
        ]

    def reset(self):
        """Reset all tracked history."""
        self.tool_history.clear()
        self.recent_searches.clear()
        self.recent_reads.clear()
