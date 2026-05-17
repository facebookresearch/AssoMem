# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.prompts import INSTRUCTION_GENERATION, INSTRUCTION_MULTITASK


class AnswerGenerator:
    """Generate answers from retrieved memory context using an LLM."""

    def __init__(self, llm_client, multitask=False):
        """
        Args:
            llm_client: LLMClient instance.
            multitask: If True, use the multi-task prompt that also
                predicts question type as a prefix.
        """
        self.llm_client = llm_client
        self.multitask = multitask

    def generate(self, question, retrieved_utterances, max_new_tokens=128):
        """Generate an answer given a question and retrieved memories.

        Args:
            question: Query string.
            retrieved_utterances: List of (text, score) tuples from retrieval.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated answer string.
        """
        memory_str = "\n".join(
            f"Memory evidence: {text}" for text, _ in retrieved_utterances
        )

        if self.multitask:
            prompt = INSTRUCTION_MULTITASK.format(memory=memory_str, question=question)
        else:
            prompt = INSTRUCTION_GENERATION.format(memory=memory_str, question=question)

        return self.llm_client.generate(prompt, max_new_tokens=max_new_tokens)

    def batch_generate(self, questions, retrieved_utterances_list, max_new_tokens=128):
        """Generate answers for a batch of questions.

        Args:
            questions: List of query strings.
            retrieved_utterances_list: List of retrieved utterance lists.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of generated answer strings.
        """
        results = []
        for question, retrieved in zip(questions, retrieved_utterances_list):
            results.append(self.generate(question, retrieved, max_new_tokens))
        return results
