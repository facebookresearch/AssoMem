# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from utils.prompts import INSTRUCTION_CLUE_TAG


@dataclass
class ClueTag:
    """A clue (topic) extracted from a single session."""

    session_id: str
    clue_text: str
    utterances: list = field(default_factory=list)


class ClueTagger:
    """Extracts a concise topic/clue per dialogue session using an LLM."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def tag_session(self, session, session_id):
        """Extract a clue from a single session.

        Args:
            session: List of turn dicts with 'content' key.
            session_id: Identifier for the session.

        Returns:
            ClueTag with the extracted topic and utterances.
        """
        session_text = "\n".join(
            turn["content"] for turn in session if "content" in turn
        )
        prompt = INSTRUCTION_CLUE_TAG.format(session=session_text)
        clue_text = self.llm_client.generate(prompt, max_new_tokens=64).strip()

        utterances = [turn["content"] for turn in session if "content" in turn]

        return ClueTag(
            session_id=session_id,
            clue_text=clue_text,
            utterances=utterances,
        )

    def tag_all_sessions(self, sessions, session_ids):
        """Extract clues from all sessions.

        Args:
            sessions: List of sessions (each a list of turn dicts).
            session_ids: List of session identifiers.

        Returns:
            List of ClueTag objects.
        """
        clue_tags = []
        for session, sid in zip(sessions, session_ids):
            clue_tags.append(self.tag_session(session, sid))
        return clue_tags
