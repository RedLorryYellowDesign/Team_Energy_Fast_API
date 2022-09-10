from typing import Optional, List
from optparse import Option
from uuid import UUID, uuid4
from pydantic import BaseModel
from enum import Enum

class Gender (str, Enum):
    male = "male"
    femail = "female"

class Role(str, Enum):
    admin = "admin"
    user = "user"
    student = "student"

class User(BaseModel):
    id: Option[UUID] = uuid4()
    first_name: str
    last_name: str
    middle_name: Optional[str]
    gender: Gender
    roles: List[Role]
