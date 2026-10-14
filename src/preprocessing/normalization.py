def normalize_intensity(image, min_value, max_value):
    normalized_image = (
        image - min_value
    ) / (max_value - min_value)

    return normalized_image
    