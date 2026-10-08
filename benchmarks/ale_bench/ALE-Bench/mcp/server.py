import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession

import ale_bench
from ale_bench.code_language import CodeLanguage, JudgeVersion
from ale_bench.result import CodeRunResult
from ale_bench.schemas import ProblemSerializable, ResultSerializable
from ale_bench.session import Session as ALEBenchSession

# Constants
MAX_SESSIONS = int(os.environ.get("ALE_BENCH_MCP_MAX_SESSIONS", "4"))
NUM_WORKERS = int(os.environ.get("ALE_BENCH_MCP_NUM_WORKERS", "13"))
LITE_VERSION = bool(int(os.environ.get("ALE_BENCH_MCP_LITE_VERSION", "0")))


@dataclass
class AppContext:
    """
    Application context for the MCP server.
    """

    ale_bench_sessions: dict[str, ALEBenchSession]


def get_current_sessions(ctx: Context[ServerSession, AppContext]) -> dict[str, ALEBenchSession]:
    """
    Retrieves the current ALE-Bench sessions from the context.
    """
    return ctx.request_context.lifespan_context.ale_bench_sessions


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Application lifespan context manager.
    Initializes the ALE-Bench sessions and provides them to the application context.
    """
    ale_bench_sessions: dict[str, ALEBenchSession] = {}
    try:
        yield AppContext(ale_bench_sessions=ale_bench_sessions)
    finally:  # FIXME: Not invoked when the server is restarted
        for session in ale_bench_sessions.values():
            await session.close()


mcp = FastMCP("ALE-Bench MCP Server", dependencies=["ale_bench"], lifespan=app_lifespan)


@mcp.resource("description://app")
async def app_description() -> str:
    """
    Returns the description of the ALE-Bench MCP application.
    This is used to provide metadata about the application.

    Returns:
        str: The description of the application.
    """
    return (
        "ALE-Bench is a benchmark for evaluating AI systems on score-based algorithmic programming contests. "
        "Drawing on real-world tasks from the AtCoder Heuristic Contest (AHC), "
        "ALE-Bench presents optimization problems (e.g., routing and scheduling) "
        "that are computationally hard and admit no known exact solution."
    )


@mcp.tool()
async def check_app() -> str:
    """
    Health check endpoint for the application.
    Returns a simple message indicating the server is running.

    Returns:
        str: A message indicating the server is running.
    """
    return f"Server `{mcp.name}` is running."


@mcp.tool()
async def list_problem_ids() -> list[str]:
    """
    Lists all available problem IDs in the ALE-Bench.
    Returns a list of problem IDs.

    Returns:
        list[str]: A list of problem IDs available in ALE-Bench.
    """
    return ale_bench.list_problem_ids(lite_version=LITE_VERSION)  # type: ignore[no-any-return]


@mcp.tool()
async def list_current_sessions() -> list[str]:
    """
    Lists all currently active ALE-Bench sessions.
    Returns a list of problem IDs.

    Returns:
        list[str]: A list containing problem IDs.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    return sorted(list(current_sessions.keys()))


@mcp.tool()
async def get_problem(problem_id: str) -> ProblemSerializable:
    """
    Retrieves the problem for the given problem ID from ALE-Bench.
    This can only be called after starting a session for the problem ID.

    Args:
        problem_id (str): The ID of the problem to retrieve.

    Returns:
        ProblemSerializable: The problem object.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ProblemSerializable.from_problem(ale_bench_session.problem)


@mcp.tool()
async def get_public_seeds(problem_id: str) -> list[int]:
    """
    Retrieves the public seeds for the given problem ID.
    This can only be called after starting a session for the problem ID.

    Args:
        problem_id (str): The ID of the problem to retrieve public seeds for.

    Returns:
        list[int]: A list of public seeds for the problem.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ale_bench_session.public_seeds  # type: ignore[no-any-return]


@mcp.tool()
async def get_rust_tool_source(problem_id: str) -> dict[str, str]:
    """
    Retrieves the Rust tool source code for the given problem ID.
    This can only be called after starting a session for the problem ID.

    Args:
        problem_id (str): The ID of the problem to retrieve the Rust tool source for.

    Returns:
        dict[str, str]: A dictionary containing the Rust tool source code.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    rust_src_dir = ale_bench_session.rust_src_dir
    rust_tool_sources = {}
    if (rust_src_dir / "lib.rs").is_file():
        with open(rust_src_dir / "lib.rs", "r") as f:
            rust_tool_sources["lib.rs"] = f.read()
    if (rust_src_dir / "bin" / "gen.rs").is_file():
        with open(rust_src_dir / "bin" / "gen.rs", "r") as f:
            rust_tool_sources["bin/gen.rs"] = f.read()
    if (rust_src_dir / "bin" / "vis.rs").is_file():
        with open(rust_src_dir / "bin" / "vis.rs", "r") as f:
            rust_tool_sources["bin/vis.rs"] = f.read()
    return rust_tool_sources


@mcp.tool()
async def get_remaining_time(problem_id: str) -> float:
    """
    Retrieves the remaining time for the given problem ID.
    This can only be called after starting a session for the problem ID.

    Args:
        problem_id (str): The ID of the problem to retrieve remaining time for.

    Returns:
        float: The remaining time in seconds.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ale_bench_session.session_remaining_time.total_seconds()  # type: ignore[no-any-return]


@mcp.tool()
async def get_visualization_server_port(problem_id: str) -> int:
    """
    Retrieves the port number for the visualization server of the given problem ID.
    This can only be called after starting a session for the problem ID.

    Args:
        problem_id (str): The ID of the problem to retrieve the visualization server port for.

    Returns:
        int: The port number of the visualization server.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    assert ale_bench_session.visualization_server_port is not None, (
        "Something went wrong: Visualization server port is not set."
    )
    return ale_bench_session.visualization_server_port  # type: ignore[no-any-return]


@mcp.tool()
async def start_session(problem_id: str) -> str:
    """
    Starts a new ALE-Bench session for the given problem ID.
    Returns the session object.

    Args:
        problem_id (str): The problem ID of the problem to start a session for.

    Returns:
        str: Confirmation message indicating the session has been started.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id in current_sessions:
        raise ValueError(f"Session for problem ID `{problem_id}` already exists.")
    if len(current_sessions) >= MAX_SESSIONS:
        raise ValueError(
            "Maximum number of sessions reached. Please close an existing session before starting a new one."
        )
    ale_bench_session = ale_bench.start(
        problem_id=problem_id,
        lite_version=LITE_VERSION,
        num_workers=NUM_WORKERS,
        run_visualization_server=True,
    )
    current_sessions[problem_id] = ale_bench_session
    return f"Session started for problem ID `{problem_id}`."


@mcp.tool()
async def code_run(
    problem_id: str,
    input_str: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
    time_limit: float | None = None,
    memory_limit: int | str | None = None,
) -> CodeRunResult:
    """
    Run arbitrary code with input and return stdout, stderr, exit status, time, and memory.

    Args:
        problem_id (str): The problem ID to generate a case for.
        input_str (str): The input string for the code run.
        code (str): The code to run.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.
        time_limit (float, optional): The time limit in seconds. Defaults to None.
        memory_limit (int | str, optional): The memory limit in bytes. Defaults to None

    Returns:
        CodeRunResult: The result of the code run.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ale_bench_session.code_run(
        input_str=input_str,
        code=code,
        code_language=code_language,
        judge_version=judge_version,
        time_limit=time_limit,
        memory_limit=memory_limit,
    )


@mcp.tool()
async def case_gen(problem_id: str, seed: list[int] | int = 0, gen_kwargs: dict[str, Any] = {}) -> list[str] | str:
    """
    Generates a case using the given seed and generation arguments for the specified problem ID.

    Args:
        problem_id (str): The problem ID to generate a case for.
        seed (list[int] | int, optional): The seed(s) for the case generation. Defaults to 0.
        gen_kwargs (dict): The generation arguments. Defaults to an empty dictionary.

    Returns:
        list[str] | str: The generated case(s). If `seed` is a list, returns a list of cases.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ale_bench_session.case_gen(seed=seed, gen_kwargs=gen_kwargs)  # type: ignore[no-any-return]


@mcp.tool()
async def case_eval(
    problem_id: str,
    input_str: list[str] | str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
    time_limit: float | None = None,
    memory_limit: int | str | None = None,
) -> ResultSerializable:
    """
    Evaluates with a given case(s) for the specified problem ID.

    Args:
        problem_id (str): The problem ID to evaluate the case for.
        input_str (list[str] | str): The input string(s) for the evaluation.
        code (str): The code to evaluate.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.
        time_limit (float, optional): The time limit in seconds. Defaults to None.
        memory_limit (int | str, optional): The memory limit in bytes. Defaults to None.

    Returns:
        ResultSerializable: The evaluation result.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ResultSerializable.from_result(
        ale_bench_session.case_eval(
            input_str=input_str,
            code=code,
            code_language=code_language,
            judge_version=judge_version,
            time_limit=time_limit,
            memory_limit=memory_limit,
            skip_local_visualization=True,
        )
    )


@mcp.tool()
async def case_vis(
    problem_id: str,
    input_str: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
    time_limit: float | None = None,
    memory_limit: int | str | None = None,
) -> Image:
    """
    Visualizes the solution with a given case for the specified problem ID.

    Args:
        problem_id (str): The problem ID to visualize the case for.
        input_str (str): The input string for the visualization.
        code (str): The code to visualize.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.
        time_limit (float, optional): The time limit in seconds. Defaults to None.
        memory_limit (int | str, optional): The memory limit in bytes. Defaults to None.

    Returns:
        Image: The visualization image of the evaluation result.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    result = ale_bench_session.case_eval(
        input_str=input_str,
        code=code,
        code_language=code_language,
        judge_version=judge_version,
        time_limit=time_limit,
        memory_limit=memory_limit,
        skip_local_visualization=False,
    )
    local_visualization = result.case_results[0].local_visualization
    buffer = io.BytesIO()
    local_visualization.save(buffer, format="webp")
    return Image(data=buffer.getvalue(), format="webp")


@mcp.tool()
async def case_gen_eval(
    problem_id: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
    seed: list[int] | int = 0,
    time_limit: float | None = None,
    memory_limit: int | str | None = None,
    gen_kwargs: dict[str, Any] = {},
) -> ResultSerializable:
    """
    Evaluates with a generated case(s) for the specified problem ID.

    Args:
        problem_id (str): The problem ID to generate and evaluate the case for.
        code (str): The code to evaluate.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.
        seed (list[int] | int, optional): The seed for the case generation. Defaults to 0.
        time_limit (float, optional): The time limit in seconds. Defaults to None.
        memory_limit (int | str, optional): The memory limit in bytes. Defaults to None.
        gen_kwargs (dict): The generation arguments. Defaults to an empty dictionary.

    Returns:
        ResultSerializable: The evaluation result.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ResultSerializable.from_result(
        ale_bench_session.case_gen_eval(
            code=code,
            code_language=code_language,
            judge_version=judge_version,
            seed=seed,
            time_limit=time_limit,
            memory_limit=memory_limit,
            gen_kwargs=gen_kwargs,
            skip_local_visualization=True,
        )
    )


@mcp.tool()
async def case_gen_vis(
    problem_id: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
    seed: int = 0,
    time_limit: float | None = None,
    memory_limit: int | str | None = None,
    gen_kwargs: dict[str, Any] = {},
) -> Image:
    """
    Visualizes the solution with a generated case(s) for the specified problem ID.

    Args:
        problem_id (str): The problem ID to generate and visualize the case for.
        code (str): The code to visualize.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.
        seed (int, optional): The seed for the case generation. Defaults to 0.
        time_limit (float, optional): The time limit in seconds. Defaults to None.
        memory_limit (int | str, optional): The memory limit in bytes. Defaults to None.
        gen_kwargs (dict): The generation arguments. Defaults to an empty dictionary.

    Returns:
        Image: The visualization image of the evaluation result.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    result = ale_bench_session.case_gen_eval(
        code=code,
        code_language=code_language,
        judge_version=judge_version,
        seed=seed,
        time_limit=time_limit,
        memory_limit=memory_limit,
        gen_kwargs=gen_kwargs,
        skip_local_visualization=False,
    )
    local_visualization = result.case_results[0].local_visualization
    buffer = io.BytesIO()
    local_visualization.save(buffer, format="webp")
    return Image(data=buffer.getvalue(), format="webp")


@mcp.tool()
async def local_visualization(
    problem_id: str,
    input_str: str,
    output_str: str,
) -> Image:
    """
    Generates a local visualization image for the given input and output strings of the specified problem ID.

    Args:
        problem_id (str): The problem ID to generate the local visualization for.
        input_str (str): The input string.
        output_str (str): The output string.

    Returns:
        Image: The local visualization image.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    local_visualization = ale_bench_session.local_visualization(
        input_str=input_str,
        output_str=output_str,
    )
    if local_visualization is None:
        raise ValueError("Local visualization failed.")
    buffer = io.BytesIO()
    local_visualization.save(buffer, format="webp")
    return Image(data=buffer.getvalue(), format="webp")


@mcp.tool()
async def public_eval(
    problem_id: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
) -> ResultSerializable:
    """
    Evaluates with pre-defined cases for the specified problem ID.

    Args:
        problem_id (str): The problem ID to evaluate the public case for.
        code (str): The code to evaluate.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.

    Returns:
        ResultSerializable: The evaluation result.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    return ResultSerializable.from_result(
        ale_bench_session.public_eval(
            code=code,
            code_language=code_language,
            judge_version=judge_version,
            skip_local_visualization=True,
        )
    )


@mcp.tool()
async def private_eval(
    problem_id: str,
    code: str,
    code_language: CodeLanguage | str,
    judge_version: JudgeVersion | str | None = None,
) -> tuple[ResultSerializable, int, int]:
    """
    Evaluates with private cases for the specified problem ID.
    This can only be called once per problem ID and the session will be closed after the evaluation.

    Args:
        problem_id (str): The problem ID to evaluate the private case for.
        code (str): The code to evaluate.
        code_language (CodeLanguage | str): The code language.
        judge_version (JudgeVersion | str, optional): The judge version. Defaults to None.

    Returns:
        ResultSerializable: The evaluation result.
        int: The rank of the submission.
        int: The performance of the submission.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions[problem_id]
    private_result, rank, performance = ale_bench_session.private_eval(
        code=code,
        code_language=code_language,
        judge_version=judge_version,
    )
    return ResultSerializable.from_result(private_result), rank, performance


@mcp.tool()
async def close_session(problem_id: str) -> str:
    """
    Closes an existing ALE-Bench session for the given problem ID.
    Returns a confirmation message.

    Args:
        problem_id (str): The problem ID of the session to close.

    Returns:
        str: Confirmation message indicating the session has been closed.
    """
    current_sessions = get_current_sessions(mcp.get_context())
    if problem_id not in current_sessions:
        raise ValueError(f"No session found for problem ID `{problem_id}`.")
    ale_bench_session = current_sessions.pop(problem_id)
    ale_bench_session.close()
    return f"Session closed for problem ID `{problem_id}`."


if __name__ == "__main__":
    mcp.run(transport="stdio")
