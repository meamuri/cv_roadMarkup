def print_help():
    pass


def print_main_menu():
    print('\n')
    print("1.\tОбработать фотографии.")
    print("\t\tТестовый набор должен находиться в папке 'dataset' каталога программы")
    print("\t\tПри этом фото с разметкой должны быть расположены в 'dataset/markup/'")
    print("\t\tПри этом фото а без нее в каталоге 'dataset/unmarkup/'")
    print("2.\tГенерировать новый набор данных на основе фотографий")
    print("\t\tРезультат сохраняется в папку 'dataset/preview'")
    print("3.\tОбучить нейросеть")
    print("4.\tСправка по программе")
    print("0.\tВыход")


def input_to_digit(user_input):
    if len(user_input) != 1 or user_input[0] < '0' or user_input[0] > '4':
        return None

    return int(user_input)
