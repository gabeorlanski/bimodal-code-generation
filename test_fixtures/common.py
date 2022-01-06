import pytest


@pytest.fixture()
def tiny_model_name():
    yield "patrickvonplaten/t5-tiny-random"
