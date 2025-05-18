import os
import subprocess
import openai
import time
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt

debugging_promptss = [
    " ",

    "Fix the bugs in this code.",
    
    "Debug this code and provide the corrected version. Don't add comments or explanations.",
    
    "Fix the code to make the failing pytest pass. The test error is included below. Return only the corrected code.",
    
    "Review the following code and fix the issues indicated by the pytest failure. Maintain the original formatting style and don't add explanatory comments.",
    
    "Analyze this code based on the pytest error message below. Make necessary corrections while preserving the original functionality. Return the complete file with your fixes.",
    
    "Debug the code based on the provided pytest failure. Focus on fixing only what's needed to make the test pass. Don't rename variables or functions unless absolutely necessary. Return only the corrected code.",
    
    "Fix the following code to resolve the pytest failure shown below. Focus on addressing the specific error while ensuring you don't introduce new bugs. Don't add additional functionality or modify unrelated parts of the code.",
    
    "Debug this code using the pytest error message as guidance. Fix all issues related to the test failure and check for similar issues elsewhere in the code. Ensure proper error handling and edge case management. Return the complete corrected program without explanations.",
    
    "Carefully analyze this code in light of the pytest failure below. Fix the immediate issue and review the rest of the code for similar problems. Maintain compatibility with the existing API and ensure all edge cases are handled. Don't add logging statements or debug code. Return the entire corrected file.",
    
    "Thoroughly debug this code based on the pytest failure. Identify and fix the root cause of the issue while ensuring compatibility with all existing functionality. Review similar patterns in the code that might cause related failures. Optimize any inefficient implementations you find, but prioritize correctness. Don't change the public API or add dependencies. Handle all potential exceptions appropriately. Return the complete corrected file without explanatory comments."
]
system_promptssss = [
    "You are a code debugging assistant. Fix bugs in the provided code.",
    
    "You are a code debugging assistant focused on efficiency. Fix bugs in the provided code and return only the corrected code without explanations.",
    
    "You are CodeFixBot, a specialized debugging assistant. Analyze the provided code and pytest error message. Fix only what's necessary to make the test pass. Return the corrected code without additional comments.",
    
    "You are a code debugging specialist. Your task is to review code and fix any issues indicated by the pytest failure. Maintain original formatting and variable names when possible. Provide only the corrected code.",
    
    "You are CodeDoctor, an expert debugging assistant. When analyzing code with pytest errors, you identify both the immediate issue and potential related problems. Make minimal necessary changes and return the complete corrected file.",
    
    "You are a code debugging expert. Your approach is methodical and conservative. Fix the issues indicated by the pytest error without changing function signatures or adding features. Only modify what's necessary to resolve the problem.",
    
    "You are CodeFixGPT, an AI specialized in debugging code. You are direct, efficient, and focus solely on solving the problem at hand. When presented with code and pytest errors, analyze thoroughly but make minimal changes. Return the entire file with your corrections.",
    
    "You are BugBuster, a code debugging specialist. Your goal is to fix issues while ensuring compatibility with existing functionality. Analyze the provided pytest errors carefully, identify the root cause, and implement the most conservative fix possible. Do not add explanatory comments unless requested.",
    
    "You are CodeMender, an AI designed for precise code fixes. You analyze code systematically, prioritizing correctness over elegance. When given pytest failures, you fix the immediate issue and check for similar problems elsewhere. You make minimal changes and never add new features or dependencies.",
    
    "You are a senior code debugging specialist with expertise across multiple programming languages. Your approach is thorough but conservative. When presented with code and pytest failures, you first understand the underlying issue completely, then implement the most targeted fix possible. You're cautious about edge cases and potential unintended consequences. You never add features or significant changes beyond what's necessary. Return the complete corrected file without explanations unless specifically requested."
]

system_prompts = [
    "You are CodeFixGPT, an AI specialized in debugging code. You are direct, efficient, and focus solely on solving the problem at hand. When presented with code and pytest errors, analyze thoroughly but make minimal changes. Return the entire file with your corrections. Do not comment the code or anything else.",    "You are BugBuster, a code debugging specialist. Your goal is to fix issues while ensuring compatibility with existing functionality. Analyze the provided pytest errors carefully, identify the root cause, and implement the most conservative fix possible. Do not add explanatory comments unless requested.",
   "You are a code-fixing assistant. Given code and a pytest failure message, fix only the specific bugs that make the test fail. Don't introduce any changes to unrelated code. Return the complete corrected code without explanations.",
    "You are a Python debugging assistant. Your task is to fix the code based on the provided pytest error. Focus on identifying and correcting the root cause of the failure. Maintain the existing code structure. Do not introduce new features or explanations. Return only the corrected file content.",
    "You are a Python debugging expert. Given a code snippet and pytest output, apply the most functional fix to resolve the failure. If the issue persists after the first fix, reanalyze the code for related bugs. Don't modify names or structure unless absolutely required. Output the full corrected code with no comments."
]
system_promptssss = [
"You are a Python debugging expert. Given a code snippet and pytest output, apply the most conservative fix to resolve the failure. If the issue persists after the first fix, reanalyze the code for related bugs. Don't modify names or structure unless absolutely required. Output the full corrected code with no comments."
]

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

BUG_FILE = "swi_test/repos/cookiecutter/cookiecutter/generate.py"
TEST_FILE = "swi_test/repos/cookiecutter/tests/test_generate.py"
LOG_FILE = "bugfix_gpt_test_log.json"

TEST_CMD = ["python3", "pybughive.py", "test", "cookiecutter-18"]
RESET_CMD = ["python3", "pybughive.py", "checkout", "cookiecutter-18"]

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def call_gpt(system_prompt, file_content, bug_message):
    test_file = read_file(TEST_FILE)
    user = f"Bug message: {bug_message} \n\nFile content:\n{file_content}" #\n\nTest content:\n{test_file}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        temperature=0.5,
    )
    return response.choices[0].message.content

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def test_passes(returncode, stdout, stderr):
    return returncode == 0 and "FAILED" not in stdout and "ERROR" not in stdout

results = []
for i, system_prompt in enumerate(system_prompts):
    print(f"\n=== System Prompt {i+1}/{len(system_prompts)} ===\n{system_prompt}\n")
    prompt_result = {"system_prompt": system_prompt, "success": False, "iterations": 0, "logs": []}
    bug_message = None  
    for attempt in range(1, 4):
        print(f"Attempt {attempt}...")
        orig_code = read_file(BUG_FILE)
        code, out, err = run_cmd(TEST_CMD)
        passed = test_passes(code, out, err)
        prompt_result["logs"].append({"iteration": attempt, "pre_fix_passed": passed, "pre_fix_stdout": out, "pre_fix_stderr": err})
        if passed:
            print("Test already passes, no fix needed.")
            prompt_result["success"] = True
            prompt_result["iterations"] = attempt
            break
        print(out, err)
        bug_message = f"Test output (stdout):\n{out}\nTest output (stderr):\n{err}"
        try:
            fixed_code = call_gpt(system_prompt, orig_code, bug_message)
            if fixed_code.strip().startswith("```"):
                lines = fixed_code.strip().split('\n')
                lines = [line for line in lines if not line.strip().startswith("```")]
                fixed_code = "\n".join(lines)
                fixed_code = '\n'.join(fixed_code.strip().split('\n')[1:])
                if fixed_code.strip().endswith('```'):
                    fixed_code = '\n'.join(fixed_code.strip().split('\n')[:-1])
        except Exception as e:
            print(f"OpenAI API error: {e}")
            prompt_result["logs"].append({"iteration": attempt, "error": str(e)})
            break
        write_file(BUG_FILE, fixed_code)
        code, out, err = run_cmd(TEST_CMD)
        passed = test_passes(code, out, err)
        prompt_result["logs"].append({
            "iteration": attempt,
            "post_fix_passed": passed,
            "post_fix_stdout": out,
            "post_fix_stderr": err,
            "model_output_file": fixed_code  #
        })
        print(f"Test passed after fix: {passed}")
        if passed:
            prompt_result["success"] = True
            prompt_result["iterations"] = attempt
            break
        run_cmd(RESET_CMD)
        time.sleep(2)
    if not prompt_result["success"]:
        prompt_result["iterations"] = 3
    results.append(prompt_result)
    run_cmd(RESET_CMD)
    time.sleep(2)

with open(LOG_FILE, "w") as f:
    json.dump(results, f, indent=2)

import numpy as np
successes = [1 if r["success"] else 0 for r in results]
iterations = [r["iterations"] for r in results]
labels = [f"Prompt {i+1}" for i in range(len(system_prompts))]
colors = ['green' if s else 'red' for s in successes]

plt.figure(figsize=(14, 7))
bars = plt.bar(labels, iterations, color=colors)
plt.title("Actual Iterations to Success/Fail per System Prompt")
plt.ylabel("Actual Number of Iterations")
plt.xlabel("System Prompt")
plt.xticks(rotation=45, ha='right')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Passed'), Patch(facecolor='red', label='Failed')]
plt.legend(handles=legend_elements, loc='upper right')

for bar, iter_count, passed in zip(bars, iterations, successes):
    label = f"{iter_count} ({'PASS' if passed else 'FAIL'})"
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("bugfix_gpt_test_results.png")
plt.show()

summary_lines = []
for i, r in enumerate(results):
    status = 'SUCCESS' if r['success'] else 'FAIL'
    summary_lines.append(f"Prompt {i+1}: {status} in {r['iterations']} iteration(s)")
    for log in r['logs']:
        if 'error' in log:
            summary_lines.append(f"  Iteration {log['iteration']}: OpenAI API error: {log['error']}")
        else:
            pre = 'PASS' if log.get('pre_fix_passed') else 'FAIL'
            summary_lines.append(f"  Iteration {log['iteration']} pre-fix: {pre}")
            if 'post_fix_passed' in log:
                post = 'PASS' if log.get('post_fix_passed') else 'FAIL'
                summary_lines.append(f"  Iteration {log['iteration']} post-fix: {post}")
summary = '\n'.join(summary_lines)
with open("bugfix_gpt_test_summary.txt", "w") as f:
    f.write(summary)
print("\nSummary written to bugfix_gpt_test_summary.txt:\n")
print(summary)
print("\nExperiment complete. Results saved to bugfix_gpt_test_log.json, bugfix_gpt_test_results.png, and bugfix_gpt_test_summary.txt.")