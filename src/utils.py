"""
utils.py
Contains helper functions and domain knowledge logic (Sailing Rules).
"""

# Mapa klas na komunikaty dla żeglarza
SAILING_ADVICE = {
    "Cumulonimbus": {
        "risk": "HIGH",
        "message": "DANGER: Storm imminent. Reef the sails immediately and close hatches.",
        "color": "red"
    },
    "Cumulus": {
        "risk": "LOW",
        "message": "Safe: Fair weather. Good conditions for full sails.",
        "color": "green"
    },
    "Cirrus": {
        "risk": "MEDIUM",
        "message": "INFO: Weather change approaching within 24h. Monitor barometer.",
        "color": "yellow"
    },
    "Stratus": {
        "risk": "LOW",
        "message": "Stable: Overcast, possible drizzle. Visibility might be reduced.",
        "color": "gray"
    }
}

def get_advice(cloud_type):
    """Returns business logic decision based on cloud classification."""
    return SAILING_ADVICE.get(cloud_type, {
        "risk": "UNKNOWN",
        "message": "Unknown cloud type. Proceed with caution."
    })