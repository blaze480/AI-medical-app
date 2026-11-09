MEDICAL_INFO = {

    "NORMAL": {

        "title": "No Pneumonia Pattern Detected",

        "description":
            "The AI model did not detect a pattern "
            "classified as pneumonia in this X-ray.",

        "causes":
            "A normal result means that the model did not "
            "identify a pneumonia pattern. Other conditions "
            "may still be present.",

        "guidance":
            "If symptoms are present, the image should still "
            "be reviewed by a qualified healthcare professional."
    },

    "PNEUMONIA": {

        "title": "Possible Pneumonia",

        "description":
            "The AI model classified the X-ray as showing "
            "a pattern associated with pneumonia.",

        "causes":
            "Pneumonia can have different causes, including "
            "bacterial, viral and other infections.",

        "guidance":
            "A healthcare professional should review the "
            "X-ray and clinical symptoms. Treatment depends "
            "on the underlying cause and clinical assessment."
    }
}