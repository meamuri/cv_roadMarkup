from .dialog import *
from .app.init import make_model_from_dataset


def general_app():
    while True:
        print_main_menu()
        scan = input("Введите пункт меню -> ")
        action = input_to_digit(user_input=scan)
        if action == 1:  # обработать
            make_model_from_dataset()
        elif action == 2:  # создать
            pass
        elif action == 3:  # обучить. Выход из функции с True!
            return True
        elif action == 4:  # справка
            print_help()
        elif action == 0:
            return False
        else:
            print("Некорректный ввод! Пожалуйста, попробуйте снова")


def next_app():
    pass


def run():
    if general_app():
        next_app()
    print("Программа завершила работу!")


    # create_dataset()
    # make_model_from_dataset()
    # just_do_it()

    # get_coefs()
    # classifier = train()
    #
    # features = data.imread('output/_11.jpg')
    #
    # classifier.predict(features)
