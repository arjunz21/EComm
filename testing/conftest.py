import pytest

################## Setup and Teardown Steps ##################
@pytest.fixture()
def setup(domain):
    print("Setup steps", domain)
    yield
    print("Teardown steps")

################## Parameterize PyTest ##################
def pytest_addoption(parser):
    parser.addoption("--domain")

@pytest.fixture()
def domain(request):
    return request.config.getoption("--domain")

################## PyTest HTML Report ##################
@pytest.mark.optionalhook 
def pytest_configure(config):
    config._metadata = {
        "Project Name": "ECOMM - emart",
        "Module Name": "EmartAPI",
        "Tester":"Arjun Gadvi"}

@pytest.mark.optionalhook
def pytest_metadata(metadata):
    metadata.pop("JAVA_HOME", None)
    # metadata.pop("Plugins", None)