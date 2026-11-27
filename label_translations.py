LABEL_BAHASA_INDONESIA = {
    "Bacterial Spot": "Bercak Bakteri",
    "Cercospora Leaf Spot": "Bercak Daun Cercospora",
    "Curl Virus": "Virus Keriting Daun",
    "Healthy Leaf": "Daun Sehat",
    "Nutrition Deficiency": "Kekurangan Nutrisi",
    "White spot": "Bercak Putih",
}


def label_to_indonesian(label):
    return LABEL_BAHASA_INDONESIA.get(label, label)
