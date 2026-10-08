import time
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# ==========================================
# 0. Core Data Structures
# ==========================================

@dataclass
class GeneratedArtifact:
    artifact_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionTrace:
    """
    Records the true execution trace of the Artifact in the environment.
    Contains only objective 'Evidence', no subjective scoring.
    """
    artifact_id: str
    is_completed: bool            # Whether execution completed (no timeout / underlying crash)
    return_code: int              # Return code (0 usually means success)
    stdout: str                   # Standard output / LLM review text
    stderr: str                   # Standard error / Error stack trace
    execution_time_ms: float      # Execution time in milliseconds
    reproduction_command: str     # Instruction for developers to reproduce locally
    metrics: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# 1. Real LLM Client (Imported)
# ==========================================

# Avoid redefining, import directly from the artifact synthesizer module
from artifact_synthesizer import BaseLLMClient
from llm_clients import MockLLMClient

# ==========================================
# 2. Harness Adapters (Strategy Pattern)
# ==========================================

class BaseHarness(ABC):
    @abstractmethod
    def run(self, artifacts: List[GeneratedArtifact]) -> List[ExecutionTrace]:
        pass

# ------------------------------------------
# Harness 1: Code Harness (Run in real environment)
# ------------------------------------------
class PythonCodeHarness(BaseHarness):
    """
    Real Python code sandbox adapter.
    Writes code to a temporary file and runs it independently using subprocess,
    collecting true objective errors.
    """
    def __init__(self, timeout_seconds: int = 5, python_executable: str = "python3"):
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable

    def run(self, artifacts: List[GeneratedArtifact]) -> List[ExecutionTrace]:
        traces = []
        for artifact in artifacts:
            diff_error = (artifact.metadata or {}).get("diff_apply_error")
            if isinstance(diff_error, str) and diff_error.strip():
                trace = ExecutionTrace(
                    artifact_id=artifact.artifact_id,
                    is_completed=True,
                    return_code=1,
                    stdout="",
                    stderr=f"[DiffApplyError] {diff_error.strip()}",
                    execution_time_ms=0.0,
                    reproduction_command="not executed: artifact rejected before sandbox",
                )
                trace.metrics["score"] = -float("inf")
                traces.append(trace)
                continue

            # --- Clean up Markdown block formatting from LLM responses (支持 cpp/rust 等多语言) ---
            from benchmark_adapter import BenchmarkTaskHarness as _BH
            artifact.content = _BH._clean_llm_output(artifact.content)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(artifact.content)
                temp_file_path = temp_file.name

            reproduction_cmd = f"{self.python_executable} {temp_file_path}"
            
            start_time = time.time()
            is_completed = False
            return_code = -1
            stdout_str = ""
            stderr_str = ""

            try:
                result = subprocess.run(
                    [self.python_executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                is_completed = True
                return_code = result.returncode
                stdout_str = result.stdout
                stderr_str = result.stderr
            except subprocess.TimeoutExpired as e:
                is_completed = False
                stdout_str = e.stdout.decode('utf-8') if e.stdout else ""
                stderr_str = f"[Timeout] Execution exceeded {self.timeout_seconds} seconds."
            except Exception as e:
                is_completed = False
                stderr_str = f"[System Error] Failed to execute harness: {str(e)}"
            finally:
                execution_time_ms = (time.time() - start_time) * 1000
                # 清理临时代码文件，避免横评下 /tmp inode 爆炸
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

            traces.append(ExecutionTrace(
                artifact_id=artifact.artifact_id,
                is_completed=is_completed,
                return_code=return_code,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time_ms=execution_time_ms,
                reproduction_command=reproduction_cmd
            ))
            
        return traces

# ------------------------------------------
# Harness 2: Benchmark Harness (LLM writes test & runs)
# ------------------------------------------
class BenchmarkHarness(BaseHarness):
    """
    Benchmark adapter.
    Steps: Call LLM to write a test program based on the current Artifact 
    and user Benchmark requirements -> Concatenate and run.
    """
    def __init__(self, benchmark_query: str, llm_client: BaseLLMClient, base_harness: Optional[BaseHarness] = None):
        self.benchmark_query = benchmark_query
        self.llm_client = llm_client
        # Relies on the real code executor to run the benchmark code underlying
        self.base_harness = base_harness or PythonCodeHarness(timeout_seconds=15)

    def run(self, artifacts: List[GeneratedArtifact]) -> List[ExecutionTrace]:
        traces = []
        for artifact in artifacts:
            # 1. Dynamically generate the benchmark script
            sys_instruction = (
                "You are an expert performance engineer. "
                "Write Python code to benchmark the provided implementation against the specified criteria. "
                "Output ONLY valid Python code that can be directly appended to the end of the user's script. "
                "Use the 'time' or 'timeit' module. Print the final metrics to stdout."
            )
            user_prompt = f"Benchmark Request: {self.benchmark_query}\n\nExisting Code:\n```python\n{artifact.content}\n```"
            
            generated_responses = self.llm_client.generate(user_prompt, sys_instruction, n=1)
            benchmark_script = generated_responses[0]
            
            # Clean up Markdown code blocks
            from benchmark_adapter import BenchmarkTaskHarness as _BH
            benchmark_script = _BH._clean_llm_output(benchmark_script)

            # 2. Concatenate original code with the benchmark code
            combined_content = f"{artifact.content}\n\n# --- LLM Generated Benchmark ---\n{benchmark_script}"
            benchmark_artifact = GeneratedArtifact(
                artifact_id=f"bench_{artifact.artifact_id}",
                content=combined_content
            )

            # 3. Run in the real sandbox
            base_traces = self.base_harness.run([benchmark_artifact])
            trace = base_traces[0]
            
            # Restore original artifact_id, and note in the reproduction command
            trace.artifact_id = artifact.artifact_id
            trace.reproduction_command = f"# Contains injected LLM benchmark code\n{trace.reproduction_command}"
            
            traces.append(trace)
            
        return traces

# ------------------------------------------
# Harness 3: Reviewer Harness (Call LLM for code review)
# ------------------------------------------
class ReviewerHarness(BaseHarness):
    """
    Reviewer adapter.
    Calls LLM as a judge or Reviewer to get opinions, recorded in stdout for downstream extraction.
    """
    def __init__(self, review_criteria: str, llm_client: BaseLLMClient):
        self.review_criteria = review_criteria
        self.llm_client = llm_client

    def run(self, artifacts: List[GeneratedArtifact]) -> List[ExecutionTrace]:
        traces = []
        for artifact in artifacts:
            start_time = time.time()
            
            sys_instruction = (
                "You are an expert code reviewer and security auditor. "
                "Please review the provided artifact against the user's specific criteria. "
                "Be highly critical, point out bugs, security flaws, and performance bottlenecks."
            )
            user_prompt = f"Review Criteria: {self.review_criteria}\n\nArtifact to review:\n```\n{artifact.content}\n```"
            
            try:
                # Call LLM to get review results
                responses = self.llm_client.generate(user_prompt, sys_instruction, n=1)
                review_feedback = responses[0]
                is_completed = True
                stderr_str = ""
            except Exception as e:
                review_feedback = ""
                is_completed = False
                stderr_str = f"LLM Reviewer Call Failed: {str(e)}"
                
            execution_time_ms = (time.time() - start_time) * 1000
            
            traces.append(ExecutionTrace(
                artifact_id=artifact.artifact_id,
                is_completed=is_completed,
                return_code=0 if is_completed else 1,
                stdout=review_feedback,  # Store the LLM's evaluation verbatim as stdout
                stderr=stderr_str,
                execution_time_ms=execution_time_ms,
                reproduction_command="N/A - LLM API Call"
            ))
            
        return traces

# ==========================================
# 3. Execution Harness Manager
# ==========================================

class ExecutionHarness:
    """
    Context manager / Dispatcher that routes artifacts to the correct environment adapter.
    """
    def __init__(self, adapter: BaseHarness):
        self.adapter = adapter

    def execute(self, artifacts: List[GeneratedArtifact]) -> List[ExecutionTrace]:
        return self.adapter.run(artifacts)

# ==========================================
# 4. MVP Test Scenarios (Real Execution Environment)
# ==========================================
if __name__ == "__main__":
    print("Real Execution Harness Manager Initiated")
    
    test_artifacts = [
        # Original test (no output)
        GeneratedArtifact(
            artifact_id="art_101", 
            content="def calculate_fibonacci(n):\n    if n <= 1: return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
        ),
        # 1. Correct code with output
        GeneratedArtifact(
            artifact_id="art_correct_output",
            content="print('Hello from the sandbox!')\nx = 10 * 2\nprint(f'Result: {x}')"
        ),
        # 2. Code with a syntax error
        GeneratedArtifact(
            artifact_id="art_syntax_error",
            content="def missing_colon()\n    print('This will cause a SyntaxError')"
        ),
        # 3. Code with an infinite loop
        GeneratedArtifact(
            artifact_id="art_infinite_loop",
            content="import time\nwhile True:\n    time.sleep(0.5)\n    print('Looping...')"
        )
    ]
    
    llm_client = MockLLMClient(canned_response="print('elapsed_seconds: 0.0')")
    
    print("\n" + "="*50)
    print("--- 1. Testing Code Harness (Real Subprocess) ---")
    print("="*50)
    code_harness = ExecutionHarness(adapter=PythonCodeHarness())
    code_traces = code_harness.execute(test_artifacts)
    for t in code_traces:
        print(f"Artifact ID: {t.artifact_id} | Completed: {t.is_completed} | Time: {t.execution_time_ms:.2f}ms")
        print(f"Reproduce via: {t.reproduction_command}")
        print(f"Errors (if any): {t.stderr}")
        
    print("\n" + "="*50)
    print("--- 2. Testing Reviewer Harness (Mock LLM Call) ---")
    print("="*50)
    reviewer_adapter = ReviewerHarness(
        review_criteria="Check if the time complexity is optimal, and if not, provide a critique.",
        llm_client=llm_client
    )
    reviewer_harness = ExecutionHarness(adapter=reviewer_adapter)
    rev_traces = reviewer_harness.execute(test_artifacts)
    print(f"Reviewer Output:\n{rev_traces[0].stdout}")
    
    print("\n" + "="*50)
    print("--- 3. Testing Benchmark Harness (LLM + Real Subprocess) ---")
    print("="*50)
    benchmark_adapter = BenchmarkHarness(
        benchmark_query="Run the function with n=30 and measure the time taken in seconds.",
        llm_client=llm_client
    )
    benchmark_harness = ExecutionHarness(adapter=benchmark_adapter)
    benchmark_traces = benchmark_harness.execute(test_artifacts)
    print(f"Benchmark Output:\n{benchmark_traces[0].stdout}")
