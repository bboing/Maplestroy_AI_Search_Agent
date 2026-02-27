"""
SQLAlchemy Base 클래스
모든 모델이 상속받을 Base 클래스 정의
"""
from sqlalchemy.orm import declarative_base

# 모든 모델이 상속받을 Base 클래스
Base = declarative_base()
