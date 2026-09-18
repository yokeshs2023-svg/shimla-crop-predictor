def calculate_frost_risk(temperature):
    """
    Calculate frost risk based on temperature.
    """

    if temperature <= 0:
        return "HIGH"

    return "LOW"


def get_frost_advisory(frost_risk):
    """
    Provide an advisory based on frost risk.
    """

    if frost_risk == "HIGH":
        return (
            "Frost risk is high. "
            "Protect frost-sensitive crops."
        )

    return (
        "Frost risk is low. "
        "Normal crop monitoring is recommended."
    )


if __name__ == "__main__":

    print("========== FROST RISK ==========")

    temperature = float(
        input("Enter temperature (°C): ")
    )

    frost_risk = calculate_frost_risk(
        temperature
    )

    advisory = get_frost_advisory(
        frost_risk
    )

    print("\nTemperature:", temperature, "°C")
    print("Frost Risk:", frost_risk)

    if frost_risk == "HIGH":
        print("❄️ Frost Risk: HIGH")

    print("Advisory:", advisory)