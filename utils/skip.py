
import json

def skip():
    try:
        return json.load(open("../.skip.json"))
    except:
        reset_skip()

def reset_skip():
    json.dump(False, open("../.skip.json", "w"))

def set_skip():
    json.dump(True, open("../.skip.json", "w"))