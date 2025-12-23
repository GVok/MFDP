from models.orm_wallet import WalletEntity
from models.orm_transaction import TransactionEntity
from models.orm_subscription import UserSubscriptionEntity


def _register_and_login(public_client, username="bill", email="bill@example.com", password="secret12") -> str:
    r = public_client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert r.status_code == 201, r.text

    r = public_client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_wallet_balance_and_topup(public_client, db_session):
    token = _register_and_login(public_client, username="w01", email="w01@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    r = public_client.get("/api/wallet/balance", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["balance_rub"] == 0

    r = public_client.post("/api/wallet/top-up", headers=headers, json={"amount_rub": 1000})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["new_balance_rub"] == 1000
    tx_id = out["transaction_id"]

    tx = db_session.query(TransactionEntity).filter(TransactionEntity.id == tx_id).first()
    assert tx is not None
    wallet = db_session.query(WalletEntity).filter(WalletEntity.user_id == tx.user_id).first()
    assert wallet is not None
    assert wallet.balance_rub == 1000


def test_billing_subscribe_insufficient_balance_returns_402(public_client):
    token = _register_and_login(public_client, username="b01", email="b01@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    r = public_client.post("/api/billing/subscribe", headers=headers, json={"plan": "standard"})
    assert r.status_code == 402, r.text
    assert "insufficient" in r.text.lower() or "balance" in r.text.lower()


def test_billing_subscribe_success_201_and_wallet_decreases(public_client, db_session):
    token = _register_and_login(public_client, username="b02", email="b02@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    r = public_client.post("/api/wallet/top-up", headers=headers, json={"amount_rub": 1000})
    assert r.status_code == 201, r.text

    r = public_client.post("/api/billing/subscribe", headers=headers, json={"plan": "standard"})
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert data["plan"]["code"] == "standard"
    assert data["plan"]["price_rub"] == 590

    sub = db_session.query(UserSubscriptionEntity).first()
    assert sub is not None
    assert sub.plan == "standard"
    assert sub.status == "active"

    wallet = db_session.query(WalletEntity).filter(WalletEntity.user_id == sub.user_id).first()
    assert wallet is not None
    assert wallet.balance_rub == 410

    txs = db_session.query(TransactionEntity).filter(TransactionEntity.user_id == sub.user_id).all()
    assert any(t.amount_rub == -590 for t in txs)


def test_billing_me_returns_plan_and_usage(public_client):
    token = _register_and_login(public_client, username="b03", email="b03@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    r = public_client.get("/api/billing/me", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["plan"]["code"] in ("freemium", "standard", "business", "pro")
    assert isinstance(data["period_yyyymm"], int)
    assert data["images_generated"] >= 0
