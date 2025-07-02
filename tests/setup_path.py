import sys
import os

def setup_project_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 이 모듈이 import될 때 자동으로 실행
setup_project_path()