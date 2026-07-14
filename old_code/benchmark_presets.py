"""Unsupported benchmark options archived during publication cleanup."""

UNWIRED_PRESETS = {
    "coord": {
        "use_coordinate_descent": True,
        "coordinate_steps": 60,
    },
    "adaptive": {
        "adaptive_coordinate_descent": True,
        "coordinate_steps": 30,
    },
}

UNWIRED_CLI_OPTIONS = ("use_logit_updates",)

# These names were never EnergyCoordinator constructor fields. Selecting either
# preset, or passing the archived CLI option, raised TypeError before a benchmark
# could start.
