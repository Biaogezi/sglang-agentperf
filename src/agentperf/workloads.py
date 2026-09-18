"""Original synthetic conversations: no user data or external tool execution."""

from __future__ import annotations


def synthetic_agent_trace(sessions: int = 24) -> dict:
    conversations = []
    for session in range(sessions):
        context = "\n".join(
            f"Module component_{j}: validates incoming records, checks field types, "
            "and returns a compact diagnostic status."
            for j in range(88)
        )
        turns = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a coding assistant reviewing a fictional "
                        "repository. Tool observations are supplied by the user; never execute tools. "
                        "Give concise engineering explanations. Do not repeat the repository listing.",
                    },
                    {
                        "role": "user",
                        "content": f"Session {session}: audit component_{session}. "
                        f"Repository context:\n{context}\nDescribe the first check you would perform.",
                    },
                ]
            }
        ]
        observations = [
            (
                f"Search result: component_{session} is called by parser.py and handler.py. "
                "The caller passes an optional integer field but its default is the string '0'. "
                "Explain the most likely type mismatch and suggest a focused unit test."
            ),
            (
                "Test result: the integer-input case passed and the missing-field case failed. "
                "The assertion expected zero of integer type, but received a string. "
                "Describe a minimal fix without changing the public function signature."
            ),
            (
                "Patch result: the default value was changed to the integer zero. "
                "Both regression tests now pass, and the repository has no changed dependencies. "
                "Summarize the fix and name one remaining edge case to check."
            ),
        ]
        for observation in observations:
            turns.append({"messages": [{"role": "user", "content": observation}]})
        conversations.append(turns)
    return {
        "metadata": {
            "kind": "original-synthetic-tool-observation-replay",
            "sessions": sessions,
            "turns": 4,
            "tool_execution": False,
            "description": "SGLang's upstream agentic-trace client feeds each real assistant "
            "response into the following turn. This is not a real SWE-bench trace or task score.",
        },
        "conversations": conversations,
    }
