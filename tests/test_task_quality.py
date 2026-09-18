from agentperf.task_quality import judge, regression_tasks


def test_regression_tasks_are_stable_and_cover_categories():
    tasks = regression_tasks()
    assert tasks == regression_tasks()
    assert len(tasks) == len({task["id"] for task in tasks}) == 40
    assert {task["category"] for task in tasks} == {
        "arithmetic",
        "json_extraction",
        "long_retrieval",
    }


def test_json_judge_requires_schema_and_integer_type():
    task = {"category": "json_extraction", "expected": {"city": "Paris", "days": 1}}
    assert judge(task, '{"days":1,"city":"Paris"}')
    assert not judge(task, '{"days":true,"city":"Paris"}')
    assert not judge(task, '{"days":1.0,"city":"Paris"}')
    assert not judge(task, '```json\n{"days":1,"city":"Paris"}\n```')
    assert not judge(task, "[]")
