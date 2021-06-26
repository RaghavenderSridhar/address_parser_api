from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
def find_all_users():
    return "LIST all user"