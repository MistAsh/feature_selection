from python_avatars import (
    HairType,
    HairColor,
    MouthType,
    NoseType,
    SkinColor,
    FacialHairType,
    AccessoryType,
    EyebrowType
)
import numpy as np

__all__ = [
    "SMALL_HAIRSTYLES",
    "MID_HAIRSTYLES",
    "LONG_HAIRSTYLES",
    "FEATURES_MAPPING",
    "MAX_LENGTH_ENCODING_LABEL",
    "TOTAL_BITS_COUNT"
]

SMALL_HAIRSTYLES = [
    HairType.SIDES,
    HairType.MOHAWK,
]

MID_HAIRSTYLES = [
    HairType.CAESAR_SIDE_PART,
    HairType.CAESAR,
    HairType.SHORT_DREADS_1,
    HairType.SHORT_CURLY,
    HairType.SHORT_FLAT,
    HairType.SHORT_ROUND,
    HairType.SHORT_WAVED,
    HairType.BUZZCUT,
]

LONG_HAIRSTYLES = [
    HairType.FRO_BAND,
    HairType.FRO,
    HairType.CURLY,
    HairType.CURLY_2,
    HairType.SHAGGY_MULLET,
    HairType.SHAGGY,
    HairType.SHORT_DREADS_2,
    HairType.STRAIGHT_1,
    HairType.STRAIGHT_STRAND,
    HairType.CORNROWS,
    HairType.DREADS,
    HairType.EINSTEIN_HAIR,
    HairType.EVIL_SPIKE,
    HairType.HALF_SHAVED,
    HairType.WILD,
]

FEATURES_MAPPING = {
    "top": {
        0: (HairType.NONE, "NO HAIR"),
        1: (lambda: np.random.choice(MID_HAIRSTYLES), "MEDIUM HAIR"),
        2: (lambda: np.random.choice(SMALL_HAIRSTYLES), "SMALL HAIR"),
        3: (lambda: np.random.choice(LONG_HAIRSTYLES), "LONG HAIR"),
    },
    "hair_color": {
        0: (HairColor.BLACK, "BLACK"),
        1: (HairColor.RED, "RED"),
        2: (HairColor.BLONDE, "BLONDE"),
        3: (HairColor.SILVER_GRAY, "SILVER GRAY")
    },
    "mouth": {
        0: (MouthType.SERIOUS, "SERIOUS"),
        1: (MouthType.SMILE, "SMILE"),
    },
    "nose": {
        0: (NoseType.WIDE, "WIDE"),
        1: (NoseType.SMALL, "SMALL")
    },
    "eyebrows": {
        0: (EyebrowType.ANGRY_NATURAL, "ANGRY NATURAL"),
        1: (EyebrowType.FLAT_NATURAL, "FLAT NATURAL")
    },
    "skin_color": {
        0: (SkinColor.TANNED, "TANNED"),
        1: (SkinColor.PALE, "PALE"),
        2: (SkinColor.BLACK, "BLACK"),
        3: (SkinColor.YELLOW, "YELLOW")
    },
    "facial_hair": {
        0: (FacialHairType.NONE, "NONE"),
        1: (FacialHairType.BEARD_LIGHT, "BEARD LIGHT"),
        2: (FacialHairType.BEARD_MEDIUM, "BEARD MEDIUM"),
        3: (FacialHairType.BEARD_MAGESTIC, "BEARD MAJESTIC")
    },
    "accessory": {
        0: (AccessoryType.NONE, "NONE"),
        1: (AccessoryType.ROUND, "ROUND")
    },
}

MAX_LENGTH_ENCODING_LABEL = np.max(
    [
        1 + len(key) + np.max([len(v) for k, (_, v) in value.items()])
        for key, value in FEATURES_MAPPING.items()
    ]
)  # FEATURE -> VALUE
TOTAL_BITS_COUNT = sum([(len(v) - 1).bit_length()
                        for v in FEATURES_MAPPING.values()])
