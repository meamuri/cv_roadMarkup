from .dialog import *
from .app.init import make_model_from_dataset
from .app.utils import create_dataset
from .app.network import show_graphics
from .app.task import make_network
from .app.processing import process_image

from skimage import data

PREDICT_FOLDER = './predict/'


def run():
    """
    Программа, вызываемая из точки входа приложения
    :return: None
    """

    def general_app():
        """
        Программа, осуществляющая подготовку всех данных для работы с нейросетью
        :return: None или классификатор, в случае вызова пользователем определенного пункта меню
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
                return make_network()
            elif action == 4:  # справка
                print_help()
            elif action == 0:
                return None
            else:
                print("\nНекорректный ввод! Пожалуйста, попробуйте снова")

    def next_app(the_classifier):
        """
        Программа, позволяющая взаимодействовать с обученной нейросетью
        :return: None
        """
        while True:
            print_sub_menu()
            scan = input("Введите пункт меню -> ")
            action = input_to_digit(user_input=scan, item_max=3)
            if action == 1:  # обработать
                work_with_network(the_classifier)
            elif action == 2:  # иллюстрировать
                show_graphics()
            elif action == 3:  # справка
                print_help_for_sub()
            elif action == 0:
                return
            else:
                print("\nНекорректный ввод! Пожалуйста, попробуйте снова")

    def work_with_network(the_classifier):
        """
        функция получает классификатор, и с его помощью
        для всех фотографий папки predict определяет, содержится ли на ней дорожная разметка
        или нет
        :param the_classifier: обученая нейронная сеть
        :return: None
        """
        import os
        all_images = [data.imread(PREDICT_FOLDER + name) for name in os.listdir(PREDICT_FOLDER)
                      if os.path.isfile(os.path.join(PREDICT_FOLDER, name))]

        for ind, img in enumerate(all_images):
            print(the_classifier)
            _ = process_image(img)
            # print(ind + 1, '\tmaybe ok')  # the_classifier.predict(img)
        print('нейронную сеть необходимо обучить!')

    classifier = general_app()
    if classifier:
        print("\nНейросеть успешно обучена!")
        print("\nПриложение по распознаванию дорожной разметки готово и запущено")
        next_app(classifier)
    print("\nПрограмма завершила работу!")
