"""Base SQLAlchemy configuration."""

from sqlalchemy import Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr


class Base:
    """Base class for all SQLAlchemy models."""

    @declared_attr.directive
    def __tablename__(cls):
        return cls.__name__.lower() + "s"

    created_at = Column(DateTime, default=func.now(), nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=True)


Base = declarative_base(cls=Base)
