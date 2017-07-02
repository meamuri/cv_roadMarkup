from .dialog import *
from .app.init import make_model_from_dataset
from .app.utils import create_dataset


def general_app():
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
    while True:
        print_sub_menu()
        scan = input("Введите пункт меню -> ")
        action = input_to_digit(user_input=scan, item_max=1)
        if action == 1:  # обработать
            pass
        elif action == 0:
            return
        else:
            print("\nНекорректный ввод! Пожалуйста, попробуйте снова")


def run():
    if general_app():
        print("\nНейросеть успешно обучена!")
        print("\nПриложение по распознаванию дорожной разметки готово и запущено")
        next_app()
    print("\nПрограмма завершила работу!")

