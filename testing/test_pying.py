import sys
import pytest

class Test_Pying:

    @pytest.mark.general
    @pytest.mark.parametrize("uname, pwd", [("admin", "admin"), ("admin", "password")])
    def test_login(self, uname, pwd):
        print("username:", uname)
        print("password:", pwd)
    
    @pytest.mark.skip
    def test_skip(self):
        print("skip")
    
    @pytest.mark.skipif(sys.version_info<(3,9), reason="Python version not supported")
    def test_skipif(self):
        print("skipif")
    
    @pytest.mark.xfail
    def test_xfail(self):
        print("xfail")
        assert False
    
    def test_1(self, setup):
        print("test 1")
        x = 10
        y = 20
        assert x!=y
    
    def test_2(self, setup):
        print("test 2")
        name = "FastAPI"
        title = "FastAPI a api development"
        assert name in title, "Title does not match"
    
    def test_3(self, setup):
        print("test 3")
        assert True
        
    @pytest.mark.general
    def test_logout(self):
        print("logout")
    