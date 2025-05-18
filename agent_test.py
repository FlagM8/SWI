import os
import time
import subprocess
import json
from typing import Dict, List, Tuple, Optional, TypedDict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

def test_passes(returncode, stdout, stderr):
    return returncode == 0 and "FAILED" not in stdout and "ERROR" not in stdout

class SelfHealingState(TypedDict, total=False):
    test_output: str
    test_error: str
    return_code: Optional[int]
    success: bool
    analysis: Any

class SelfHealingAgent:
    def __init__(self, project: str, bug_id: str, folder_path: str, pybughive_path: str, openai_api_key: str = None):
        """Initialize the self-healing agent.
        
        Args:
            project: Project identifier
            bug_id: Bug identifier
            folder_path: Path to the folder containing the project files
            pybughive_path: Path to the pybughive.py test runner
            openai_api_key: OpenAI API key (will use environment variable if not provided)
        """
        self.project = project
        self.bug_id = bug_id
        self.folder_path = folder_path
        self.pybughive_path = pybughive_path
        self.test_command = f"python3 {pybughive_path} test {project}-{bug_id}"
        self.max_iterations = 10
        self.metrics = {
            "iterations": 0,
            "start_time": None,
            "end_time": None,
            "success": False,
            "fixes_applied": []
        }
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3
        )
        
        # Set up the LangGraph
        self.setup_graph()

    def setup_graph(self):
        """Set up the LangGraph for the self-healing workflow."""
        workflow = StateGraph(state_schema=SelfHealingState)
        
        workflow.add_node("run_test", self.run_test)
        workflow.add_node("analyze_error", self.analyze_error)
        workflow.add_node("fix_issue", self.fix_issue)
        workflow.add_node("check_iterations", self.check_iterations)
        
        workflow.add_conditional_edges(
            "run_test",
            lambda x: "check_iterations" if x.get("success") else "analyze_error"
        )
        workflow.add_edge("analyze_error", "fix_issue")
        workflow.add_edge("fix_issue", "run_test")
        workflow.add_conditional_edges(
            "check_iterations",
            lambda x: END if (x.get("success") or self.metrics["iterations"] >= self.max_iterations) else "analyze_error"
        )
        
        workflow.set_entry_point("run_test")
        
        self.graph = workflow.compile()

    def run_test(self, state: Dict) -> Dict:
        """Run the test command and capture the output."""
        current_iteration = self.metrics["iterations"] + 1
        self.metrics["iterations"] = current_iteration
        
        print(f"\n--- Iteration {current_iteration} ---")
        print(f"Running test: {self.test_command}")
        
        result = subprocess.run(
            self.test_command, 
            shell=True, 
            cwd=self.folder_path,
            capture_output=True, 
            text=True
        )
        
        state["test_output"] = result.stdout
        state["test_error"] = result.stderr
        state["return_code"] = result.returncode
        state["success"] = test_passes(result.returncode, result.stdout, result.stderr)
        
        if state["success"]:
            print("Test passed successfully!")
            self.metrics["success"] = True
        else:
            print(f"Test failed with return code {result.returncode}")
            if result.stdout:
                print(f"\nStandard Output:\n{result.stdout}")
            if result.stderr:
                print(f"\nError Output:\n{result.stderr}")
        
        return state

    def analyze_error(self, state: Dict) -> Dict:
        """Analyze the error output to determine the issue."""
        #error_text = state["test_error"] if state["test_error"] else state["test_output"]
        error_text = state["test_output"]
        
        prompt = PromptTemplate.from_template(
            """You are debugging a Python program. Given the error output, identify:
            1. The file(s) that needs to be modified (tests are correct, issue is in the code)
            2. The specific issue(s) in the code
            3. A clear description of what needs to be fixed
            
            Error output:
            {error_text}
            
            Respond ONLY with valid JSON. Do not include markdown formatting or any explanation.
            Your JSON must have the following keys:
            - files_to_modify: list of filepaths
            - issues: list of issue descriptions
            - fix_description: string explaining what to fix
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        response_text = chain.invoke({"error_text": error_text})
        if response_text.strip().startswith("```"):
            lines = response_text.strip().split('\n')
            lines = [line for line in lines if not line.strip().startswith("```")]
            response_text = "\n".join(lines)
            response_text = '\n'.join(response_text.strip().split('\n')[1:])
            if response_text.strip().endswith('```'):
                response_text = '\n'.join(response_text.strip().split('\n')[:-1])
        print(f"\nResponse from LLM:\n{response_text}")

        try:
            analysis = json.loads(response_text)
            state["analysis"] = analysis
            print(f"\nAnalysis:\n{json.dumps(analysis, indent=2)}")
        except json.JSONDecodeError:
            print("Couldn't parse analysis as JSON, extracting information manually")
            files_mentioned = []
            for line in error_text.split("\n"):
                if ".py" in line:
                    potential_file = line.split(".py")[0] + ".py"
                    potential_file = potential_file.split("/")[-1]
                    if os.path.exists(os.path.join(self.folder_path, potential_file)):
                        files_mentioned.append(potential_file)
            
            state["analysis"] = {
                "files_to_modify": list(set(files_mentioned)),
                "issues": ["Error parsing issue from output"],
                "fix_description": error_text
            }
            print(f"\nExtracted files to check: {files_mentioned}")
        
        return state

    def read_file(self, filepath: str) -> str:
        """Read the content of a file."""
        full_path = os.path.join(self.folder_path, filepath)
        try:
            with open(full_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return ""

    def write_file(self, filepath: str, content: str) -> None:
        """Write content to a file."""
        full_path = os.path.join(self.folder_path, filepath)
        try:
            with open(full_path, 'w') as file:
                file.write(content)
            print(f"Updated file: {filepath}")
        except Exception as e:
            print(f"Error writing to file {filepath}: {e}")

    def fix_issue(self, state: Dict) -> Dict:
        """Fix the issue in the identified files."""
        analysis = state["analysis"]
        files_to_modify = analysis["files_to_modify"]
        fix_description = analysis["fix_description"]
        
        for file_path in files_to_modify:
            current_content = self.read_file(file_path)
            if not current_content:
                continue
            
            prompt = PromptTemplate.from_template(
                """You are fixing a bug in a Python file.
                
                File path: {file_path}
                Bug description: {fix_description}
                
                Current file content:
                ```python
                {current_content}
                ```
                
                Provide ONLY the corrected content of the file, with no additional explanation.
                Keep your changes minimal - fix only what's needed to resolve the issue.
                Return the ENTIRE file content, not just the changed part.
                """
            )
            
            chain = prompt | self.llm | StrOutputParser()
            fixed_content = chain.invoke({
                "file_path": file_path,
                "fix_description": fix_description,
                "current_content": current_content
            })
            lines = fixed_content.strip().split('\n')
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            fixed_content = "\n".join(lines)
            
            if fixed_content != current_content:
                self.write_file(file_path, fixed_content)
                self.metrics["fixes_applied"].append({
                    "file": file_path,
                    "iteration": self.metrics["iterations"]
                })
            else:
                print(f"No changes needed for {file_path}")
        
        return state

    def check_iterations(self, state: Dict) -> Dict:
        """Check if we should continue or end the process."""
        return state

    def run(self):
        """Run the self-healing agent."""
        self.metrics["start_time"] = time.time()
        
        # Initial state
        state = {
            "test_output": "",
            "test_error": "",
            "return_code": None,
            "success": False,
            "analysis": None
        }
        
        # Run the graph
        try:
            self.graph.invoke(state)
        except Exception as e:
            print(f"Error running graph: {e}")
        
        self.metrics["end_time"] = time.time()
        self.print_metrics()

    def print_metrics(self):
        """Print the metrics collected during the run."""
        runtime = self.metrics["end_time"] - self.metrics["start_time"]
        
        print("\n" + "="*50)
        print("Self-Healing Agent Metrics")
        print("="*50)
        print(f"Project: {self.project}, Bug ID: {self.bug_id}")
        print(f"Total iterations: {self.metrics['iterations']}")
        print(f"Total runtime: {runtime:.2f} seconds")
        print(f"Average time per iteration: {runtime/max(1, self.metrics['iterations']):.2f} seconds")
        print(f"Success: {self.metrics['success']}")
        print(f"Fixes applied: {len(self.metrics['fixes_applied'])}")
        for fix in self.metrics['fixes_applied']:
            print(f"  - Iteration {fix['iteration']}: {fix['file']}")
        print("="*50)


if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Self-Healing Agent")
    # parser.add_argument("project", help="Project identifier")
    # parser.add_argument("bug_id", help="Bug identifier")
    # parser.add_argument("folder_path", help="Path to the folder containing the project files")
    # parser.add_argument("pybughive_path", help="Path to the pybughive.py test runner")
    # parser.add_argument("--api-key", help="OpenAI API key")
    
    # args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    agent = SelfHealingAgent(
        project="discord.py",
        bug_id="7818",
        folder_path="/Users/jcmuchar/Documents/pybughive/swi_test/repos/discord.py/",
        # project="cookiecutter",
        # bug_id="18",
        # folder_path="/Users/jcmuchar/Documents/pybughive/swi_test/repos/cookiecutter",
        pybughive_path="/Users/jcmuchar/Documents/pybughive/pybughive.py",
        openai_api_key=os.getenv("OPENAI_API_KEY") )
    
    agent.run()