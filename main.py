# Fast API for the main app
# ---| ALL IMPORTS |---
from typing import List
from uuid import UUID, uuid4
from fastapi import FastAPI, HTTPException
from models import Calling_Forcast_Model

app = FastAPI()


# ---| API END POINT |---
# These are all the API Keys that can be called.
@app.get("/")
def root():
    return {"message": "Hello World. This is the Team Energy API. Please use the API keys below to call the API."}

# ---| Calls to see of API is working |---
@app.get("/all_good")
def check():
    return {"API Up and Running"}

# take an input from the user
@app.post("/api/v1/model_1")
async def model_1(Calling_Forcast_Model: Calling_Forcast_Model):
    db.append()


# # ---| DATABASE OF USERS |---
# db: List[User] = [
#     User(
#         id=UUID("249eee0f-f6ef-414b-b6c9-5caef388f533"),
#         first_name="John",
#         last_name="Doe",
#         gender = Gender.female,
#         roles = [Role.student]
#     ),
#     User(
#         id=UUID("c7aabefe-0af2-490a-9fc8-1fb8d9ac96bb"),
#         first_name="Alex",
#         last_name="Smith",
#         gender = Gender.male,
#         roles = [Role.admin, Role.user]
#     )
# ]

# ---| API END POINT |---
# These are all the API Keys that can be called.
@app.get("/")
def root():
    return {"message": "Hello World magic"}

@app.get("/api/v1/users")
async def fetch_users():
    return db;

@app.post("/api/v1/users")
async def create_user(user: User):
    db.append(user)
    return {"id": user.id}

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: UUID):
    for user in db:
        if user.id == user_id:
            db.remove(user)
            return
    raise HTTPException(
        status_code=404,
        detail=f"User with id {user_id} not found")

@app.put("/api/v1/users/{user_id}")
async def update_user(user_update: User_Update_Requests, user_id: UUID):
    for user in db:
        if user.id == user_id:
            if user_update.first_name is not None:
                user.first_name = user_update.first_name
            if user_update.last_name is not None:
                user.last_name = user_update.last_name
            if user_update.middle_name is not None:
                user.middle_name = user_update.middle_name
            if user_update.roles is not None:
                user.roles = user_update.roles
            return
    raise HTTPException(
        status_code=404,
        details=f"User with id: {user_id} not found"
        )
