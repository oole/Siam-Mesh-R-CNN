from datetime import datetime
from termcolor import colored


def date_print(message: str):
    print(colored(str(datetime.now()) + ": " + message, "green"))


def hint_print(message: str):
    print(colored(str(datetime.now()) + ": " + message, "yellow"))
