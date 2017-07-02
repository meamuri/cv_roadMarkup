# строка-справка о работе с нейросетью
HELP_FOR_NEURAL_NETWORK = \
    "Исполнение пункта меню 1 (распознавание иозбражений)" \
    "Возьмет все файлы каталоги './predict/' и отдаст на анализ нейронной сети" \
    "Для каждого файла будет получен результат" \
    "Пункт меню 2 представляет графики анализа гистограмм изображений с разметкой" \
    "и без нее."


def print_help_for_sub():
    """
    Печать подсказки для работы с нейросетью
    :return: None
    """
    print(HELP_FOR_NEURAL_NETWORK)


def print_help():
    """
    Функция печатает информацию о созданном приложении и о том, как им пользоваться
    :return: None
    """
    print('\n')
    print('Курсовая работа студента 3 курса ВГУ, ПММ, МОиАИС')
    print('Реализованные модули:')
    print('\t* обработки фото для выделения существенных признаков')
    print('\t  (применяется черно-белый фильтр, выделение краев)')
    print('\t* генерации дополнительного набора данных')
    print('\t  (применяются преобразования к исходным фотографиям)')
    print('\t* обучение и работа с нейросетью')
    print('\t  (на основе заданного набора)')
    print('Также реализован консольный инфтерфейс взаимодействия с пользователем CLI.')


def print_main_menu():
    """
    Функция печатает основное меню
    :return:
    """
    print('\n')
    print("1.\tОбработать фотографии.")
    print("\tТестовый набор должен находиться в папке 'dataset' каталога программы")
    print("\tпри этом фото с разметкой должны быть расположены в 'dataset/markup/'")
    print("\tа фотографии без нее в каталоге 'dataset/unmarkup/'")
    print("2.\tГенерировать новый набор данных на основе фотографий.")
    print("\tРезультат сохраняется в папку 'dataset/preview/'")
    print("3.\tОбучить нейросеть")
    print("4.\tСправка по программе")
    print("0.\tВыход")


def print_sub_menu():
    """
    Функция печатает меню подприложения
    :return:
    """
    print('\n')
    print("1.\tРаспознать есть ли дорожная разметка на фото")
    print("2.\tИллюстрировать графики")
    print("3.\tСправка")
    print("0.\tВыход")


def input_to_digit(user_input, item_max):
    """
    функция проверяет, что введено число, соответствующее пункту меню
    :param user_input: пользовательский ввод
    :param item_max: максимальный пункт меню в диалоге,
        больше которого пункт меню не должен быть
    :return:
    """
    if len(user_input) != 1 or user_input[0] < '0' or user_input[0] > '9':
        return -1

    res = int(user_input)
    if res > item_max:
        res = -1

    return res
