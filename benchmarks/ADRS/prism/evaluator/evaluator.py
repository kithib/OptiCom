import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
from dataclasses import dataclass

GPU_MEM_SIZE = 80 # GB
MIN_INT = float('-inf')  # Define MIN_INT as negative infinity

@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def case_score_from_kvpr(max_kvpr: float) -> float:
    """
    Convert a per-case KV cache pressure into a bounded quality score.

    Lower KVPR is better, and the transformed score is always in [0, 1].
    Failed cases are handled separately as exactly 0.
    """
    kvpr = safe_float(max_kvpr)
    if kvpr < 0:
        return 0.0
    return float(np.clip(1.0 / (1.0 + kvpr), 0.0, 1.0))


def verify_gpu_mem_constraint(placement_data: dict[int, list[Model]]) -> bool:
    """
    Verify the whether models can fit into GPU memory
    """
    # Check if the placement data is valid
    if placement_data is None:
        return False

    # Check if the placement data is valid
    for gpu_id, models in placement_data.items():
        if sum(model.model_size for model in models) > GPU_MEM_SIZE:
            return False

    return True


def calculate_kvcache_pressure(placement_data: dict[int, list[Model]]) -> float:
    """
    Calculate the KVCache pressure
    """
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)

    return max_kvpr


def generate_test_gpu_models(num_tests=50):
    """
    Generate multiple test signals with different characteristics
    """
    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num*2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(model_name=f"model_{j}", model_size=model_size, req_rate=req_rate, slo=slo, cur_gpu_id=j))

        test_cases.append((gpu_num, gpu_models))

    return test_cases

def evaluate(program_path):
    """
    Main evaluation function that tests the signal processing algorithm
    on multiple test signals and calculates the composite performance metric.
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if required function exists
        if not hasattr(program, "compute_model_placement"):
            return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": "Missing compute_model_placement function",
                }

        # Generate test gpu and models
        test_gpu_models = generate_test_gpu_models()

        # Collect metrics across all tests. Every test case contributes exactly
        # one bounded score: success -> 1/(1+KVPR), failure -> 0.
        all_kvpr = []
        case_scores = []
        all_metrics = []
        successful_runs = 0
        total_cases = len(test_gpu_models)
        failure_reasons = {}

        for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
            start_time = time.time()
            try:
                # Run the algorithm with timeout

                # Call the program's main function
                result = run_with_timeout(
                    program.compute_model_placement,
                    kwargs={
                        'gpu_num': gpu_num,
                        'models': gpu_models
                    },
                    timeout_seconds=10
                )

                execution_time = time.time() - start_time

                # Validate result format
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict, got {type(result).__name__}")

                # Validate all models are placed
                placed_models = []
                for gpu_id, assigned_models in result.items():
                    if not isinstance(assigned_models, list):
                        raise ValueError(
                            f"GPU {gpu_id} value must be list, got {type(assigned_models).__name__}"
                        )
                    placed_models.extend(assigned_models)

                if len(placed_models) != len(gpu_models):
                    raise ValueError(f"Not all models placed: {len(placed_models)}/{len(gpu_models)}")

                # Check for duplicate placements (by object identity)
                placed_ids = [id(m) for m in placed_models]
                if len(set(placed_ids)) != len(placed_ids):
                    raise ValueError("Duplicate models detected")

                # Check placed models are the exact input objects
                original_ids = {id(m) for m in gpu_models}
                if set(placed_ids) != original_ids:
                    raise ValueError("Placed models don't match input models (missing or foreign models)")

                # Verify GPU memory constraints
                if not verify_gpu_mem_constraint(result):
                    raise ValueError("GPU memory constraint violated")

                # Calculate metrics using the generated test signal
                max_kvpr = calculate_kvcache_pressure(result)
                case_score = case_score_from_kvpr(max_kvpr)

                # Store metrics
                metrics = {
                    'max_kvpr': safe_float(max_kvpr),
                    'case_score': safe_float(case_score),
                    'execution_time': safe_float(execution_time),
                    'passed': 1.0,
                }

                all_kvpr.append(safe_float(max_kvpr))
                case_scores.append(case_score)
                all_metrics.append(metrics)
                successful_runs += 1

            except TimeoutError:
                failure_reasons["Timeout"] = failure_reasons.get("Timeout", 0) + 1
                case_scores.append(0.0)
                all_metrics.append({
                    'max_kvpr': 0.0,
                    'case_score': 0.0,
                    'execution_time': safe_float(time.time() - start_time),
                    'passed': 0.0,
                    'error': 'Timeout',
                })
            except Exception as e:
                reason = str(e)[:160] or type(e).__name__
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                case_scores.append(0.0)
                all_metrics.append({
                    'max_kvpr': 0.0,
                    'case_score': 0.0,
                    'execution_time': safe_float(time.time() - start_time),
                    'passed': 0.0,
                    'error': str(e),
                })

        # Calculate aggregate metrics
        avg_kvpr = np.mean(all_kvpr) if all_kvpr else 0.0
        avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
        success_rate = successful_runs / total_cases if total_cases else 0.0
        combined_score = np.mean(case_scores) if case_scores else 0.0
        print(
            "Prism aggregate: "
            f"combined_score={safe_float(combined_score):.6f}, "
            f"success_rate={safe_float(success_rate):.3f}, "
            f"successful_runs={successful_runs}/{total_cases}, "
            f"avg_successful_kvpr={safe_float(avg_kvpr):.6f}"
        )
        if failure_reasons:
            failure_summary = "; ".join(
                f"{count}x {reason}" for reason, count in sorted(failure_reasons.items())
            )
            print(f"Prism failed cases: {failure_summary}")

        return {
                "max_kvpr": safe_float(avg_kvpr),
                "execution_time": safe_float(avg_execution_time),
                "success_rate": safe_float(success_rate),
                "combined_score": safe_float(combined_score),
                "avg_case_score": safe_float(combined_score),
                "successful_runs": safe_float(successful_runs),
                "total_cases": safe_float(total_cases),
                "metric_version": "prism_bounded_case_average_v2",
            }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": str(e)
            }


if __name__ == "__main__":
    # Backwards-compat: bridges old evaluate() -> dict to the container JSON
    # protocol.  wrapper.py is auto-injected at build time from
    # skydiscover/evaluation/wrapper.py.
    from wrapper import run

    run(evaluate)
