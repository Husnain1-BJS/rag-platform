"""Evaluation dataset for RAG system - 3 question/ground_truth pairs for quick testing."""
EVAL_EXAMPLES = [
    {
        "question": "What is CVE-2021-44228?",
        "ground_truth": "CVE-2021-44228 is Log4Shell, a critical remote code execution vulnerability in Apache Log4j 2.x versions prior to 2.17.1."
    },
    {
        "question": "What is the MITRE ATT&CK technique T1003?",
        "ground_truth": "T1003 is OS Credential Dumping in MITRE ATT&CK, involving techniques to obtain account login credentials from the operating system."
    },
    {
        "question": "What vulnerability is CVE-2023-34362?",
        "ground_truth": "CVE-2023-34362 is a critical SQL injection vulnerability in Progress Software MOVEit Transfer, exploited by the Cl0p ransomware group."
    },
]