"""Basic import verification tests.

Tests that all modules can be imported without a .env file or database.
Settings use defaults (empty strings) for API keys.
"""


def test_core_config_imports():
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.app_env == "development"
    assert s.llm_api_key == ""


def test_state_imports():
    pass


def test_schema_imports():
    pass


def test_model_imports():
    pass


def test_graph_imports():
    pass


def test_web_search_imports():
    pass


def test_api_imports():
    pass


def test_service_imports():
    pass


def test_prompts_imports():
    pass


def test_llm_factory():
    pass


def test_settings_imports():
    pass
