from agent.orchestrator import Orchestrator, _final_exception
from agent.schemas import ExecResult, Issue, RunState
from agent.schemas import TestResult as Result


def result(stderr: str = "", stdout: str = "", exit_code: int = 1) -> Result:
    return Result(passed=exit_code == 0,
                  exec=ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr, duration_s=0.1))


def tb(last: str) -> str:
    return f'Traceback (most recent call last):\n  File "repro_issue.py", line 3, in <module>\n    main()\n{last}\n'


def check(res: Result) -> tuple[str | None, bool]:
    exc = _final_exception(res)
    return exc, Orchestrator._repro_valid(res, exc)[0]


def test_improperly_configured_is_invalid():
    assert check(result(tb("ImproperlyConfigured: Requested setting"))) == ("ImproperlyConfigured", False)


def test_dotted_exception_is_invalid():
    res = result(tb("django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS"))
    assert check(res) == ("ImproperlyConfigured", False)


def test_custom_exception_without_error_suffix_is_valid():
    assert check(result(tb("sympy.core.Unsupported: cannot do this"))) == ("Unsupported", True)


def test_exception_without_message():
    assert check(result(tb("__main__.Timeout"))) == ("Timeout", True)


def test_assertion_error_is_valid():
    assert check(result(tb("AssertionError: expected 1 got 2"))) == ("AssertionError", True)


def test_no_traceback_is_invalid():
    res = result("something went wrong\nError: not a python traceback\n", stdout="hello\n")
    exc, valid = check(res)
    assert exc is None and not valid
    assert "without a traceback" in Orchestrator._repro_valid(res, exc)[1]


def test_last_traceback_wins_over_chained_one():
    text = tb("KeyError: 'a'") + "\nDuring handling of the above exception, another exception occurred:\n\n" \
        + tb("NameError: name 'x' is not defined")
    assert check(result(text)) == ("NameError", False)


def test_multiline_message_uses_exception_line():
    assert check(result(tb("AssertionError: first line\nValueError: quoted in the message"))) == ("AssertionError", True)


def test_exception_like_text_in_stdout_is_ignored():
    res = result(tb("AssertionError"), stdout="NameError: printed by the script\n")
    assert check(res) == ("AssertionError", True)


def test_syntax_error_has_no_traceback_header():
    text = '  File "/testbed/repro_issue.py", line 2\n    foo(\n       ^\nSyntaxError: invalid syntax\n'
    assert check(result(text)) == ("SyntaxError", False)


def test_trimmed_output_lost_the_header():
    text = '...[truncated]...\n  File "a.py", line 1, in f\n    f()\n  [Previous line repeated 996 more times]\nRecursionError: maximum recursion depth exceeded\n'
    assert check(result(text)) == ("RecursionError", True)


def state(**kw) -> RunState:
    issue = Issue(instance_id="a__b-1", repo="a/b", base_commit="x", version="1", problem_statement="p")
    return RunState(run_id="r", issue=issue, **kw)


def test_regression_only_flag():
    assert not state(repro_script="assert 0").regression_only
    assert state(repro_script="", verification_mode="regression_only").regression_only
    assert state(repro_script="").regression_only  # saved before verification_mode existed
