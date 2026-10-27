"""TRANSFERFIT configuration."""
from pathlib import Path

RAW_DIR = Path(r"C:\Football\DATA\project\data\raw")
PROJECT_DIR = Path(r"C:\Football\DATA\project")

DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
SRC_DIR = PROJECT_DIR / "src"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
MODELS_SAVED_DIR = PROJECT_DIR / "models_saved"
APP_DIR = PROJECT_DIR / "app"

for d in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, TABLES_DIR,
          OUTPUT_MODELS_DIR, MODELS_SAVED_DIR):
    d.mkdir(parents=True, exist_ok=True)

LALIGA_COMPETITION_ID = 795
LALIGA_AREA_ID = 724
RM_TEAM_ID_A = 675  # Real Madrid in teams.parquet (namespace A)

SUPER_SEASONS = None

# Namespace A (parquet universe)
FILES = {
    "areas": RAW_DIR / "areas.parquet",
    "players": RAW_DIR / "players.parquet",
    "player_careers": RAW_DIR / "player_careers.parquet",
    "player_season_stats": RAW_DIR / "player_season_stats.parquet",
    "pss_total": RAW_DIR / "player_season_stats_total.parquet",
    "pss_per90": RAW_DIR / "player_season_stats_per90.parquet",
    "pss_average": RAW_DIR / "player_season_stats_average.parquet",
    "pss_percent": RAW_DIR / "player_season_stats_percent.parquet",
    "player_valuations": RAW_DIR / "player_valuations.parquet",
    "teams": RAW_DIR / "teams.parquet",
    "competitions": RAW_DIR / "competitions.parquet",
    "seasons": RAW_DIR / "seasons.parquet",
    "super_seasons": RAW_DIR / "super_seasons.parquet",
}

# Namespace B (Transfermarkt-style ledger universe)
FILES_B = {
    "club_transfers": RAW_DIR / "club_transfers.csv",
    "valuations_full": RAW_DIR / "player_valuations_full.csv",
    "club_financial_metrics": RAW_DIR / "club_financial_metrics.csv",
    "team_dossier": RAW_DIR / "team_full_dossier.csv",
    "sd_db": RAW_DIR / "SD-DB(PLAYERS).csv",
}

# Optional / auxiliary
FILES_X = {
    "events_sample": RAW_DIR / "15946 (1).json",
    "rm_la_fabrica": RAW_DIR / "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026.xlsx",
    "laliga_zip": RAW_DIR / "laliga_2022_2025.zip",
}

RAW_SNAPSHOT_FILES = list(FILES.values()) + list(FILES_B.values())