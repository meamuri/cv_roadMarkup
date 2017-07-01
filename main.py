from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from src.utils import generate_dataset


def main():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    generate_dataset("markup/", datagen=datagen)
    generate_dataset("unmarkup/", datagen=datagen)


if __name__ == "__main__":
    main()