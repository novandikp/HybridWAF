import os
from signature.SignatureDatabase import SignatureDatabase

database_path = os.path.join(os.getcwd(), "database", "signatures.db")
signature_database = SignatureDatabase(database_path)


def detect_signature(signature):
    return signature_database.detect(signature)


def add_signature(signature, malicious=False):
    signature_database.add_signature(signature, malicious)


def remove_database():
    if os.path.exists(database_path):
        global signature_database
        signature_database.disconnect()
        os.remove(database_path)
        signature_database = SignatureDatabase(database_path)
