from src.init import make_model_from_dataset
from src.network import just_do_it
from src.task import train
# from src.utils import create_dataset


def main():
    # create_dataset()
    # make_model_from_dataset()
    # just_do_it()
    classifier = train()
    print("meeeh")


if __name__ == "__main__":
    main()
