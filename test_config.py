import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name=="nimfa":
        pytest.skip("Skipping test for nimfa")
    

def check_test_solver_run(solver_class):
    if solver_class.name=="nimfa":
        pytest.skip("Skipping test for nimfa")

# def check_test_dataset_get_data(dataset_class): #skipping datasets is not allowed rn
#     if dataset_class.name=="Indian Pines HSI":
#         pytest.skip("Skipping test for Indian Pines HSI")