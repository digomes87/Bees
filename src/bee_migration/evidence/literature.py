"""Figures summarising the cited studies on climate-driven bee range shifts.

Values were entered by hand as a reading of the sources listed in the README
(UFPR, Dryad, Ecology Letters, PMC); they are not extracted from the studies'
datasets.
"""

REGIONS = ("North", "Northeast", "Center-West", "Southeast", "South")

# UFPR study of 18 species in south-eastern South America
SPECIES_BY_REGION = {
    "Current (2020)": dict(zip(REGIONS, (12, 8, 15, 18, 22), strict=True)),
    "Projected (2040)": dict(zip(REGIONS, (8, 5, 12, 20, 28), strict=True)),
}
MIGRATION_DIRECTIONS = {
    "South": 45,
    "Southwest": 40,
    "Southeast": 10,
    "No Change": 3,
    "North": 2,
}
IMPACT_FACTORS = {
    "Temperature": 9,
    "Precipitation": 7,
    "Land Use": 8,
    "Monocultures": 6,
}
MIGRATION_INTENSITY = {2024: 0, 2029: 15, 2034: 35, 2039: 60, 2044: 80, 2049: 100}

# Local projections set against published ones
TEMPERATURE_INCREASE = {
    "North\n(Local Data)": 3.2,
    "Center\n(Local Data)": 2.8,
    "South\n(Local Data)": 2.1,
    "UFPR\nProjection": 2.5,
    "Aegean\nProjection": 3.5,
}
TIMELINE_YEARS = (1960, 1990, 2000, 2010, 2020, 2024, 2030, 2040, 2050)
ABUNDANCE_TIMELINE = {
    "Bombus bellicosus (UFPR)": (100, 80, 60, 40, 20, 0, 0, 0, 0),
    "General Abundance (Local)": (100, 95, 90, 85, 75, 70, 60, 50, 40),
}
PRESENT_YEAR = 2024
CRITICAL_WARMING = {
    "Eusocial\nSpecies": 2.0,
    "Solitary\nSpecies": 2.5,
    "Generalist\nSpecies": 1.8,
    "Specialist\nSpecies": 3.0,
}
AVERAGE_CRITICAL_WARMING = 2.5
BIOME_VULNERABILITY = {
    "Amazon": 7.5,
    "Cerrado": 8.2,
    "Atlantic Forest": 9.1,
    "Caatinga": 6.8,
    "Pampa": 8.8,
}

# Synthesis across studies
SOUTHWARD_SHARE_BY_STUDY = {
    "UFPR\n(18 spp)": 85,
    "Dryad\n(Neotropical)": 90,
    "North America\n(Bombus)": 70,
    "Aegean\n(Islands)": 60,
}
SCENARIO_YEARS = tuple(range(2024, 2051, 3))
MIGRATION_SCENARIOS = {
    "Conservative Scenario": (0, 10, 25, 40, 55, 70, 80, 85, 90),
    "Moderate Scenario": (0, 15, 35, 55, 70, 80, 88, 92, 95),
    "Accelerated Scenario": (0, 20, 45, 65, 80, 90, 95, 98, 100),
}
FUNCTIONAL_GROUPS = (
    "Eusocial\nBees",
    "Solitary\nBees",
    "Generalist\nPollinators",
    "Specialist\nPollinators",
)
FUNCTIONAL_RESPONSE = {
    "Migration Rate (%)": dict(zip(FUNCTIONAL_GROUPS, (75, 85, 65, 95), strict=True)),
    "Adaptation Capacity (%)": dict(
        zip(FUNCTIONAL_GROUPS, (60, 40, 80, 20), strict=True)
    ),
}
PRIORITY_AREAS = (
    "Southern\nBrazilian\nFields",
    "Atlantic\nForest\nRemnants",
    "Cerrado\nTransition",
    "Coastal\nPlains",
    "Mountain\nCorridors",
)
PRIORITY_SCORES = dict(zip(PRIORITY_AREAS, (9.5, 8.8, 8.2, 7.5, 9.0), strict=True))
SPECIES_RICHNESS = dict(zip(PRIORITY_AREAS, (45, 38, 42, 28, 35), strict=True))
