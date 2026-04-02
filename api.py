import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'budget.db')}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, index=True)
    type = Column(String, index=True)  # income or expense (lowercase in the app)
    amount = Column(Float, nullable=False)
    category = Column(String, index=True, nullable=False)
    date = Column(String, index=True, nullable=False)  # stored as YYYY-MM-DD string
    note = Column(Text, default="")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Transaction(BaseModel):
    id: Optional[str] = Field(default=None)
    type: str
    amount: float
    category: str
    date: str  # YYYY-MM-DD
    note: Optional[str] = ""

    class Config:
        orm_mode = True


app = FastAPI(title="Budget Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/transactions", response_model=List[Transaction])
def get_transactions(db: Session = Depends(get_db)):
    """Get everything, newest dates first."""
    records = (
        db.query(TransactionDB)
        .order_by(TransactionDB.date.desc(), TransactionDB.id.desc())
        .all()
    )
    return records


@app.put("/transactions/bulk", response_model=List[Transaction])
def replace_transactions(transactions: List[Transaction], db: Session = Depends(get_db)):
    """
    Wipes the table and saves this list — same idea as when I used to overwrite the whole JSON file.
    """
    # clean up the incoming list and skip bad rows
    normalized: List[TransactionDB] = []
    ids_seen = set()

    for t in transactions:
        if t.type not in {"income", "expense"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid type for transaction {t.id or 'N/A'}",
            )

        amount = abs(float(t.amount))
        if amount == 0:
            continue

        tx_id = t.id or str(uuid.uuid4())

        category = (t.category or "").strip() or "Other"
        note = (t.note or "").strip()
        date_str = t.date

        if tx_id in ids_seen:
            continue
        ids_seen.add(tx_id)

        db_obj = TransactionDB(
            id=tx_id,
            type=t.type,
            amount=amount,
            category=category,
            date=date_str,
            note=note,
        )
        normalized.append(db_obj)

    # Clear table then insert normalized set
    db.query(TransactionDB).delete()
    for obj in normalized:
        db.add(obj)
    db.commit()

    records = (
        db.query(TransactionDB)
        .order_by(TransactionDB.date.desc(), TransactionDB.id.desc())
        .all()
    )
    return records


@app.post("/transactions", response_model=Transaction, status_code=status.HTTP_201_CREATED)
def add_transaction(transaction: Transaction, db: Session = Depends(get_db)):
    """Add one row — main app uses bulk replace but I left this here if I need it."""
    if transaction.type not in {"income", "expense"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="type must be 'income' or 'expense'.",
        )

    amount = abs(float(transaction.amount))
    if amount == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="amount must be greater than zero.",
        )

    tx_id = transaction.id or str(uuid.uuid4())

    db_obj = TransactionDB(
        id=tx_id,
        type=transaction.type,
        amount=amount,
        category=(transaction.category or "").strip() or "Other",
        date=transaction.date,
        note=(transaction.note or "").strip(),
    )

    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


@app.delete("/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Delete one row by its id string."""
    tx = db.query(TransactionDB).filter(TransactionDB.id == transaction_id).first()
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )
    db.delete(tx)
    db.commit()
    return

