INSTRUCTION_LST_MEMORY = '''
### TASK DESCRIPTION
You are a helpful assistant that helps users to organize their memory records. 
Given user historical dialogues, please distinguish each utterance into long-term and short-term types based on the following description of each type.
Short-term memory: serves as a buffer that stores everything happens recently such as "had sushi yesterday" etc.
Long-term memory: serves as a user base that records long-impact information such as "chronic disease, user preference" etc.

### INPUT
Dialogue session: {session}

### OUTPUT REQUIREMENT
Generate a dict consisting of two keys: short-term memory, long-term memory in JSON format.
Each item correponds to a list of utterances. 
Please only output the dict and nothing else.
'''

INSTRUCTION_GENERATION = '''
### TASK DESCRIPTION
You are a helpful assistant that answers user's question. In this sense, you will have access to user's memory records which contain user's historical informaion.
Please note you will need to identify if the memories are useful or not for you to respond the query. 
If the memories are useful then answer the question based on the memories, otherwise answer the question based on your knowledge or answer "IDK".

### INPUT
User memory: {memory}
User query: {question}

### OUTPUT REQUIREMENT
Output the answer to the question only. Not matter you use the memory or not, please only output the answer and nothing else.
'''

INSTRUCTION_TOPIC_MEMORY = """
### TASK DESCRIPTION
You are a helpful assistant that helps users to organize their memory records. 
Given user historical dialogues, please summarize the utterances of each session with each user relevant information included.

### INPUT
Dialogue session: {session}

### OUTPUT REQUIREMENT
Generate a list such as python list consisting of summarized sentences.
Please only output the list and nothing else.
"""

INSTRUCTION_TOPIC = """
### TASK DESCRIPTION
You are a helpful assistant that helps users to organize their memory records. Next, you'll help me in organzing a user's memory records.
Given a user historical dialogue session, please summarize the session into a concise topic summary without key information lost. 
Output the topic summary sentence.

### INPUT
Dialogue session: {session}

### OUTPUT REQUIREMENT
Generate a topic summary for the given session.
Please only output the topic summary and nothing else.
"""