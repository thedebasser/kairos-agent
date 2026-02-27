"""Kairos Agent — Video Editor Agent Response Models.

Pydantic models used as Instructor response_model for LLM calls
in the Video Editor Agent pipeline. These define the structured output
schema that LLMs must conform to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HookCaptionResponse(BaseModel, frozen=True):
    """Structured output from the Caption Writer LLM (Claude Sonnet).

    The Caption Writer receives the concept brief and generates a
    compelling hook caption for the first 0-2 seconds of the video.
    POC: Hook caption only — future expansion for rule/tension/payoff.
    """

    hook_text: str = Field(
        description=(
            "A short, compelling hook caption (<=6 words) that creates "
            "curiosity or intrigue about the physics simulation. "
            "Examples: 'What if gravity reversed?', "
            "'Watch every ball find its place', "
            "'One push starts a chain reaction'"
        ),
        max_length=50,
    )
    reasoning: str = Field(
        description=(
            "Brief explanation of why this hook was chosen and "
            "what viewer emotion it targets"
        ),
    )

    @field_validator("hook_text")
    @classmethod
    def hook_text_max_words(cls, v: str) -> str:
        """Hook text must be <=6 words."""
        word_count = len(v.strip().split())
        if word_count > 6:
            msg = f"Hook text must be <=6 words, got {word_count}"
            raise ValueError(msg)
        return v.strip()


class VideoTitleResponse(BaseModel, frozen=True):
    """Structured output from the Title Writer LLM (Llama 3.1 8B local).

    The Title Writer generates a YouTube/TikTok-optimised title for
    the final video. Titles should be attention-grabbing and
    platform-appropriate.
    """

    title: str = Field(
        description=(
            "A short, engaging video title for YouTube Shorts / TikTok. "
            "Should be descriptive, intriguing, and under 80 characters. "
            "Do NOT use clickbait or misleading claims. "
            "Examples: 'When 500 Balls Meet a Marble Funnel', "
            "'The Most Satisfying Domino Chain You'll See Today'"
        ),
        max_length=100,
    )
    description: str = Field(
        default="",
        description=(
            "A brief video description (1-2 sentences) for the platform. "
            "Include relevant keywords naturally."
        ),
        max_length=300,
    )

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Title must not be empty or whitespace-only."""
        if not v or not v.strip():
            msg = "Title must not be empty"
            raise ValueError(msg)
        return v.strip()
