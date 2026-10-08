from unittest.mock import call, patch

from utils.compiler import compileAndLog


def test_compile_and_log_cleans_before_building(tmp_path):
    with patch("utils.compiler.subprocess.run") as run:
        success, error = compileAndLog(str(tmp_path))

    assert success is True
    assert error == ""
    assert run.call_args_list == [
        call(
            "make clean",
            shell=True,
            check=True,
            cwd=str(tmp_path),
            stdout=run.call_args_list[0].kwargs["stdout"],
            stderr=-2,
        ),
        call(
            "make main",
            shell=True,
            check=True,
            cwd=str(tmp_path),
            stdout=run.call_args_list[1].kwargs["stdout"],
            stderr=-2,
        ),
    ]

