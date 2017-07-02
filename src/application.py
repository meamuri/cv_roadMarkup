from .dialog import *
from .app.init import make_model_from_dataset
from .app.utils import create_dataset
from .app.network import show_graphics


def general_app():
    """
    Программа, осуществляющая подготовку всех данных для работы с нейросетью
    :return: None
    """
    while True:
        print_main_menu()
        scan = input("Введите пункт меню -> ")
        action = input_to_digit(user_input=scan, item_max=4)
        if action == 1:  # обработать
            make_model_from_dataset()
        elif action == 2:  # создать
            create_dataset()
        elif action == 3:  # обучить. Выход из функции с True!
            return True
        elif action == 4:  # справка
            print_help()
        elif action == 0:
            return False
        else:
            print("\nНекорректный ввод! Пожалуйста, попробуйте снова")


def next_app():
    """
    Программа, позволяющая взаимодействовать с обученной нейросетью
    :return: None
    """
    while True:
        print_sub_menu()
        scan = input("Введите пункт меню -> ")
        action = input_to_digit(user_input=scan, item_max=3)
        if action == 1:  # обработать
            pass
        if action == 2:  # иллюстрировать
            show_graphics()
        if action == 3:  # справка
            print_help_for_sub()
        elif action == 0:
            return
        else:
            print("\nНекорректный ввод! Пожалуйста, попробуйте снова")


def run():
    """
    Программа, вызываемая из точки входа приложения
    :return: None
    """
    if general_app():
        print("\nНейросеть успешно обучена!")
        print("\nПриложение по распознаванию дорожной разметки готово и запущено")
        next_app()
    print("\nПрограмма завершила работу!")
