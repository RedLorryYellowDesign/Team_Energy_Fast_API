# Fast API for the main app
from typing import List
from uuid import uuid4
from fastapi import FastAPI
from models import User, Gender, Role

app = FastAPI()

db: List[User] = [
        User(id=uuid4(),
        first_name="John",
        last_name="Doe",
        gender = Gender.female,
        roles = [Role.student]
    ),
        User(id=uuid4(),
        first_name="Alex",
        last_name="Smith",
        gender = Gender.male,
        roles = [Role.admin, Role.user]
    )
]

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/api/v1/users")
async def fetch_users():
    return db;
