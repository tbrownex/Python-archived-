import os

DATA_FILENAME = os.path.expanduser("~/jobNumber.txt")

def setJob(next_id):
    with open(DATA_FILENAME, "w") as fd:
        fd.write(str(next_id) + "\n")

def getJob():
    with open(DATA_FILENAME, "r") as fd:
        return fd.readline().strip()