import io

import cairosvg
import numpy as np
from PIL import Image
from python_avatars import ClothingType
from python_avatars import Avatar
from feature_selection.visualization.avatars.constants import *


def convert_vectors_to_image(vectors):
    images = []
    for vector in vectors:
        image = convert_to_avatar(vector, two_bits_encoding=True)
        image = from_avatar_to_image(image)
        images.append(image)
    return images

def encode_feature(vector: np.ndarray) -> np.ndarray:
    n = vector.size

    if n < TOTAL_BITS_COUNT:
        vector = np.pad(vector, (0, TOTAL_BITS_COUNT - n))

    new_vector = np.zeros(len(FEATURES_MAPPING))
    current_index = 0

    for index, (_, values) in enumerate(FEATURES_MAPPING.items()):
        bits_num = (len(values) - 1).bit_length()
        new_vector[index] = np.packbits(
            vector[current_index: current_index + bits_num],
            bitorder='little',
        )[0]
        current_index += bits_num
    return new_vector


def get_mapping_values(vector: np.ndarray) -> dict:
    result = {}
    feature_names = {}

    for index, (feature_name, values) in enumerate(FEATURES_MAPPING.items()):
        value_tuple = FEATURES_MAPPING[feature_name][vector[index]]

        if callable(value_tuple[0]):
            actual_value = value_tuple[0]()
            result[feature_name] = actual_value
            # Store the general category name (e.g., "MEDIUM HAIR")
            feature_names[feature_name] = value_tuple[1]
        else:
            result[feature_name] = value_tuple[0]
            feature_names[feature_name] = value_tuple[1]

    if "hair_color" in result:
        result["facial_hair_color"] = result["hair_color"]
        feature_names["facial_hair_color"] = feature_names["hair_color"]

    # Store feature names for later use if needed
    result["_feature_names"] = feature_names

    return result


def convert_to_avatar(
        vector: np.ndarray,
        elite: bool = False,
        two_bits_encoding: bool = False,
        print_features: bool = False
):
    if two_bits_encoding:
        vector = encode_feature(vector)

    if vector.size < len(FEATURES_MAPPING):
        vector = np.pad(
            vector,
            (0, len(FEATURES_MAPPING)),
            "constant"
        )
    if vector.size > len(FEATURES_MAPPING):
        vector = vector[:len(FEATURES_MAPPING)]

    kwargs = get_mapping_values(vector)

    # Print feature names if requested
    if print_features:
        feature_names = kwargs.pop("_feature_names")
        print("Avatar features:")
        for feature, name in feature_names.items():
            if feature != "facial_hair_color":  # Skip duplicated info
                print(f"  {feature}: {name}")
    else:
        # Remove the feature names from kwargs before creating Avatar
        kwargs.pop("_feature_names", None)

    if elite:
        kwargs['clothing'] = ClothingType.BOND_SUIT
        clothing_name = "BOND SUIT"
    else:
        kwargs['clothing'] = ClothingType.COLLAR_SWEATER
        clothing_name = "COLLAR SWEATER"

    if print_features:
        print(f"  clothing: {clothing_name}")

    return Avatar(**kwargs)

def from_avatar_to_image(avatar):
    avatar_svg = avatar.render()
    png_data = cairosvg.svg2png(bytestring=avatar_svg.encode("utf-8"))
    avatar_img = Image.open(io.BytesIO(png_data))
    return avatar_img

def feature_count(bits: int):
    """Calculate how many features can be encoded with the given number of bits."""
    n = 0
    bits_used = 0
    for values in FEATURES_MAPPING.values():
        bit_for_features = (len(values) - 1).bit_length()
        if bits_used + bit_for_features <= bits:
            bits_used += bit_for_features
            n += 1
        else:
            break
    return n