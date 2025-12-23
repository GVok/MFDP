import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.base import Base

import models.orm_user
import models.orm_ml_request
import models.orm_ml_task
import models.orm_prediction
import models.orm_subscription
import models.orm_usage
import models.orm_wallet
import models.orm_transaction

from main import app
from api.deps import get_db, get_current_user
from models.orm_user import UserEntity


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture()
def db_session(engine):
    connection = engine.connect()
    transaction = connection.begin()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    db = TestingSessionLocal()

    nested = connection.begin_nested()

    from sqlalchemy import event

    @event.listens_for(db, "after_transaction_end")
    def restart_savepoint(session, trans):
        nonlocal nested
        if nested.is_active:
            return
        if connection.closed:
            return
        nested = connection.begin_nested()

    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session):
    def _get_db_override():
        yield db_session

    user = UserEntity(
        username="testuser",
        email="test@example.com",
        password_hash="x",
        is_admin=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    def _get_current_user_override():
        return user

    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[get_current_user] = _get_current_user_override

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()

@pytest.fixture()
def public_client(db_session):
    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
