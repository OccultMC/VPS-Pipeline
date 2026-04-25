def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits the real Apple Look Around API")
