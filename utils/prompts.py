# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

INSTRUCTION_CLUE_TAG = """### TASK DESCRIPTION
You are a helpful assistant that helps users to organize their memory records.
Given a user historical dialogue session, please summarize the session into a concise topic summary without key information lost.
Output the topic summary sentence.

### INPUT
Dialogue session: {session}

### OUTPUT REQUIREMENT
Generate a topic summary for the given session.
Please only output the topic summary and nothing else."""

INSTRUCTION_LST_MEMORY = """### TASK DESCRIPTION
You are a helpful assistant that helps users to organize their memory records.
Given user historical dialogues, please distinguish each utterance into long-term and short-term types based on the following description of each type.
Short-term memory: serves as a buffer that stores everything happens recently such as "had sushi yesterday" etc.
Long-term memory: serves as a user base that records long-impact information such as "chronic disease, user preference" etc.

### INPUT
Dialogue session: {session}

### OUTPUT REQUIREMENT
Generate a dict consisting of two keys: short-term memory, long-term memory in JSON format.
Each item corresponds to a list of utterances.
Please only output the dict and nothing else."""

INSTRUCTION_GENERATION = """### TASK DESCRIPTION
You are a helpful assistant that answers user's questions. You will have access to user's memory records which contain user's historical information.
Please note you will need to identify if the memories are useful or not for you to answer the query.
If the memories are useful then answer the question based on the memories, otherwise answer the question based on your knowledge or answer "IDK".

### INPUT
User memory: {memory}
User query: {question}

### OUTPUT REQUIREMENT
Output the answer to the question only. No matter you use the memory or not, please only output the answer and nothing else."""

INSTRUCTION_MULTITASK = """### TASK DESCRIPTION
You are a helpful assistant that answers user's questions. You will have access to user's memory records which contain user's historical information.
First, identify the question type from [knowledge, event, temporal, preference, counterfactual].
Then, answer the question based on the provided memories.

### INPUT
User memory: {memory}
User query: {question}

### OUTPUT REQUIREMENT
Output in the format:
[TYPE: <question_type>] <answer>"""

INSTRUCTION_JUDGE = """### TASK DESCRIPTION
You are a judge that evaluates the quality of generated answers compared to golden answers.
Given a question, a generated answer, and a golden answer, determine if the generated answer is correct.

### INPUT
Question: {question}
Generated Answer: {generated}
Golden Answer: {golden}

### OUTPUT REQUIREMENT
Output only "correct" or "incorrect"."""

INSTRUCTION_JUDGE_WIN_RATE = """### TASK DESCRIPTION
You are a judge that compares two answers to a question and determines which one is better.

### INPUT
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

### OUTPUT REQUIREMENT
Output only "A" or "B" to indicate the better answer."""
