import os
import sqlite3


class SignatureDatabase:
    def __init__(self, filename: str):
        if not os.path.exists(filename):
            open(filename, "w+").close()
        self.conn = sqlite3.connect(filename)
        self.c = self.conn.cursor()
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS signatures (signature INTEGER PRIMARY KEY, malicious INTEGER)"""
        )
        self.conn.commit()

    def add_signature(self, signature, malicious: bool = False):
        hash_signature = hash(tuple(signature))
        self.c.execute(
            "INSERT INTO signatures (signature, malicious) VALUES (?, ?)",
            (hash_signature, int(malicious)),
        )
        self.conn.commit()

    def detect(self, signature):
        hash_signature = hash(tuple(signature))
        self.c.execute(
            "SELECT malicious FROM signatures WHERE signature=?", (hash_signature,)
        )
        result = self.c.fetchone()
        if result:
            return bool(result[0])
        return None

    def disconnect(self):
        self.conn.close()
