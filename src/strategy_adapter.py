import json
import re
from typing import List, Dict, Any, Optional

# ==========================================
# 0. Import Real Dependencies
# ==========================================
# We now import the actual modules we built previously
from artifact_synthesizer import BaseLLMClient
from operator_library import OperatorLibrary, Operator
from optimization_controller import RuleBasedController
from optimization_memory import OptimizationMemory, OperatorStats

# ==========================================
# 1. Strategy Adapter Implementation
# ==========================================

class StrategyAdapter:
    """
    Module ⑫: Strategy Adapter (Slow-timescale strategy regulator).
    Triggered during long stagnation, repeated failures, or regime shifts.
    Reads from the real OptimizationMemory and modifies the real Controller and OperatorLibrary.
    """
    
    def __init__(self, 
                 llm_client: BaseLLMClient, 
                 controller: RuleBasedController, 
                 operator_library: OperatorLibrary):
        self.llm_client = llm_client
        self.controller = controller
        self.operator_library = operator_library

        # Ensure Controller has a configuration object for dynamic tuning
        # Since our initial RuleBasedController didn't expose a config dict natively,
        # we dynamically attach one to store hyperparameters that the Controller's rules can read.
        if not hasattr(self.controller, 'config'):
            self.controller.config = {
                "branch_width_multiplier": 1,
                "exploration_rate": 0.2,
            }
        self.controller.config.setdefault("operator_weights", {})
        self.controller.config.setdefault("disabled_operators", [])
        self.controller.config.setdefault("preferred_operator", None)

    def adapt(self, memory: OptimizationMemory, stagnation_level: int) -> Dict[str, Any]:
        """
        Main entry point for adaptation. Called by the outer loop when the inner loop stalls.
        Returns a summary report of the adaptations made.
        """
        report = {
            "controller_adjustments": [],
            "operator_weight_adjustments": [],
            "new_operators_introduced": []
        }

        # --- Responsibility 1: Adjust Controller Preferences ---
        adj_c = self._adjust_controller_preferences(stagnation_level)
        report["controller_adjustments"].extend(adj_c)

        # --- Responsibility 2: Adjust Operator Library Weights ---
        adj_w = self._adjust_operator_weights(memory)
        report["operator_weight_adjustments"].extend(adj_w)

        # --- Responsibility 3: Introduce New Operator via LLM ---
        adj_o = self._introduce_new_operator(memory)
        if adj_o:
            report["new_operators_introduced"].append(adj_o)

        return report

    def _adjust_controller_preferences(self, stagnation_level: int) -> List[str]:
        """Dynamically tunes the controller's hyperparameters based on stagnation severity."""
        adjustments = []
        
        if stagnation_level >= 3:
            # Increase exploration and branch width significantly
            self.controller.config["branch_width_multiplier"] = 3
            self.controller.config["exploration_rate"] = 0.8
            self.controller.stagnation_threshold += 1 # Give it more patience
            adjustments.append("Increased branch_width_multiplier to 3 for broader exploration.")
            adjustments.append("Increased exploration_rate to 0.8.")
        elif stagnation_level > 0:
            self.controller.config["branch_width_multiplier"] = 2
            self.controller.config["exploration_rate"] = 0.5
            adjustments.append("Increased branch_width_multiplier to 2.")

        return adjustments

    def _adjust_operator_weights(self, memory: OptimizationMemory) -> List[str]:
        """
        Penalizes or disables operators that have a history of 0% success rate
        after a significant number of attempts, based on REAL memory stats.
        """
        adjustments = []
        
        for op_name, stat in memory.operator_stats.items():
            op = self.operator_library.get_operator(op_name)
            if not op:
                continue
                
            # Initialize custom strategy attributes dynamically if not present
            if not hasattr(op, 'weight'): op.weight = 1.0
            if not hasattr(op, 'is_active'): op.is_active = True
            
            # Rule: If an operator failed more than 3 times with 0 successes, punish its weight
            if stat.attempts >= 3 and stat.success_rate == 0.0:
                op.weight *= 0.1
                self.controller.config["operator_weights"][op_name] = op.weight
                adjustments.append(f"Penalized operator '{op_name}' weight to {op.weight:.2f} due to 0% success rate over {stat.attempts} attempts.")
                if stat.attempts >= 5:
                    op.is_active = False
                    disabled = self.controller.config.setdefault("disabled_operators", [])
                    if op_name not in disabled:
                        disabled.append(op_name)
                    adjustments.append(f"Disabled operator '{op_name}' entirely after {stat.attempts} consecutive failures.")
                    
            # Rule: Reward highly successful operators
            elif stat.attempts >= 2 and stat.success_rate > 0.5:
                op.weight *= 1.5
                self.controller.config["operator_weights"][op_name] = op.weight
                adjustments.append(f"Boosted operator '{op_name}' weight to {op.weight:.2f} due to high success rate ({stat.success_rate*100:.0f}%).")

        return adjustments

    def _introduce_new_operator(self, memory: OptimizationMemory) -> Optional[str]:
        """
        Uses the LLM to read the memory summary and synthesize a brand new,
        highly targeted Operator to break out of the current local optima.
        """
        sys_instruction = (
            "You are a Meta-Optimization Strategist. The system is stuck in a local optima. "
            "Based on the global summary of recent failures, invent a NOVEL optimization operator (strategy) "
            "that takes a fundamentally different approach.\n\n"
            "Format your response strictly as a JSON object with the following keys:\n"
            '{"name": "...", "description": "...", "instruction_template": "..."}'
        )
        
        # Read the real global LLM summary from memory
        global_summary = getattr(memory, 'global_llm_summary', 'No summary available.')
        user_prompt = f"--- Memory Summary of Stagnation ---\n{global_summary}\n\nGenerate the new operator JSON:"
        
        try:
            responses = self.llm_client.generate(user_prompt, sys_instruction, n=1)
            raw_response = responses[0]
            
            # Extract JSON from potential Markdown blocks
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if not json_match:
                return "Failed to parse JSON from LLM response."
                
            op_data = json.loads(json_match.group(0))
            
            # Ensure safe naming
            new_op_name = "dynamic_" + re.sub(r'[^a-z0-9_]', '', op_data.get("name", "custom_op").lower()[:20])
            raw_template = str(op_data.get("instruction_template", "Rewrite the artifact."))
            safe_template, rejection_reason = self._sanitize_dynamic_operator_template(raw_template)
            if rejection_reason:
                return f"Rejected new operator '{new_op_name}': {rejection_reason}"
            
            # Create a real Operator instance
            new_operator = Operator(
                name=new_op_name,
                description=op_data.get("description", "Dynamically generated operator."),
                technique_line="LLM-Synthesized",
                instruction_template=safe_template,
            )
            # Inject strategy properties
            new_operator.weight = 2.0 
            new_operator.is_active = True
            
            # Register to the REAL library
            self.operator_library.register(new_operator)
            self.controller.config["operator_weights"][new_op_name] = new_operator.weight
            self.controller.config["preferred_operator"] = new_op_name
            
            return f"Introduced new operator: '{new_op_name}' - {new_operator.description}"
            
        except Exception as e:
            return f"Error introducing new operator: {str(e)}"

    @staticmethod
    def _sanitize_dynamic_operator_template(template: str) -> tuple[str, Optional[str]]:
        """
        Dynamic operators are allowed to describe a rewrite strategy, but their
        templates must still ask the LLM to output the target artifact itself.
        Reject templates that accidentally preserve the meta-operator JSON task.
        """
        text = (template or "").strip() or "Rewrite the artifact."
        lowered = text.lower()
        meta_markers = [
            "instruction_template",
            "operator json",
            "operator spec",
            "new operator json",
            "generate the new operator",
            "required keys",
            "name, description",
            "format your response strictly as a json",
            "output json strictly",
        ]
        if any(marker in lowered for marker in meta_markers):
            return text, (
                "instruction_template describes a meta-operator/JSON response "
                "instead of an artifact rewrite."
            )

        hard_contract = (
            "\n\nHard output contract: Apply this strategy to the CURRENT ARTIFACT. "
            "Output ONLY the complete updated artifact in the target format. "
            "Do not output JSON, operator specs, markdown fences, or explanations."
        )
        if "hard output contract" not in lowered:
            text = text.rstrip() + hard_contract
        return text, None

# ==========================================
# 2. MVP Test Scenarios (Using Real Dependencies)
# ==========================================
if __name__ == "__main__":
    print("Initiating Strategy Adapter with REAL dependencies...")
    
    class MockClient:
        def generate(self, p, s, n=1):
            return ["""{
                "name": "Chunked Processing",
                "description": "Breaks large arrays into smaller chunks to avoid OOM and Timeouts.",
                "instruction_template": "Rewrite the target function to use a batching/chunking strategy."
            }"""]

    llm_client = MockClient()

    # 2. Initialize Real Subsystems
    real_controller = RuleBasedController()
    real_operator_lib = OperatorLibrary() # This now contains 'local_revision', 'error_repair', etc.
    real_memory = OptimizationMemory()

    # 3. Inject simulated failure stats into the REAL memory to trigger adaptation
    real_memory.global_llm_summary = "The system has been stuck failing with 'TimeoutError' and 'OOM' when processing large arrays. 'local_revision' has failed 5 times in a row."
    
    # Inject real stat objects into memory
    real_memory.operator_stats["local_revision"] = OperatorStats(attempts=5, successes=0)
    real_memory.operator_stats["recombination"] = OperatorStats(attempts=2, successes=2)

    # 4. Initialize Strategy Adapter
    adapter = StrategyAdapter(
        llm_client=llm_client,
        controller=real_controller, 
        operator_library=real_operator_lib
    )

    print("\n--- 1. System State BEFORE Adaptation ---")
    print(f"Controller Config: {adapter.controller.config}")
    # Print the state of two operators from the real library
    for op_name in ["local_revision", "recombination"]:
        op = real_operator_lib.get_operator(op_name)
        weight = getattr(op, 'weight', 1.0)
        active = getattr(op, 'is_active', True)
        print(f"Operator [{op_name}]: weight={weight:.2f}, active={active}")

    # Simulate triggering the adapter due to a stagnation level of 3
    print("\n🚨 Stagnation Detected! Triggering Strategy Adapter...")
    report = adapter.adapt(memory=real_memory, stagnation_level=3)

    print("\n--- 2. Adaptation Report ---")
    print(json.dumps(report, indent=2))

    print("\n--- 3. System State AFTER Adaptation ---")
    print(f"Controller Config: {real_controller.config}")
    
    # Loop over all registered operators in the real library to see the changes
    for op_name in real_operator_lib.list_available_operators():
        op = real_operator_lib.get_operator(op_name)
        weight = getattr(op, 'weight', 1.0)
        active = getattr(op, 'is_active', True)
        
        if not active:
            print(f"Operator [{op_name}]: DISABLED ❌")
        elif weight > 1.0:
            print(f"Operator [{op_name}]: BOOSTED 🚀 (weight={weight:.2f})")
        elif op_name.startswith("dynamic_"):
            print(f"Operator [{op_name}]: NEW INJECTION ✨ (weight={weight:.2f})")
        else:
            print(f"Operator [{op_name}]: weight={weight:.2f}, active={active}")
