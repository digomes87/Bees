# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `bee_migration` package under `src/` with `climate`, `analysis`, `prediction`,
  `evidence`, `charts` and `reports` modules
- `bee-migration` command (`analyze`, `predict`, `charts`, `all`) with `--data`,
  `--root` and `--seed` options
- Test suite under `tests/`
- Browser dashboard under `docs/`: the package runs in the page through Pyodide
  and draws trends, correlations, model accuracy and forecasts
- Field data chapter with observed temperatures (Open-Meteo, ERA5) and the mean
  latitude of bee records (GBIF), parsed by `bee_migration.sources`
- `CI` workflow (lint and tests) and `Pages` workflow (site deployment from `main`)
- Git Flow branching guide in `CONTRIBUTING.md`

### Changed
- Models are evaluated on the most recent years, held out chronologically, and
  tuned with time-ordered cross-validation
- Feature importance is measured by permutation for every model
- Climate stress index is expressed in regional standard deviations
- Project metadata, pytest and ruff configuration consolidated in `pyproject.toml`

### Fixed
- Prediction features no longer include the same-year change of the target
- Lagged correlations now compare temperature with bee abundance in later years
- A missing `--data` file is reported as an error instead of silently falling
  back to simulated data
- Compatibility with pandas 3

### Removed
- Trained model files from version control; `bee-migration predict` recreates them
- Simulated seasonality figures and percentile clipping of observations
- Generated literature reports; the figures they quoted live in
  `bee_migration.evidence.literature`

## [1.0.0] - 2024-01-21

### Added
- Complete English translation of README.md
- English version of all scientific charts and visualizations
- New `generate_charts_english.py` script for generating charts with English text
- Comprehensive bee migration analysis with climate change impact assessment
- Machine learning models for migration prediction (Random Forest, Gradient Boosting, SVR, Linear Regression)
- Scientific paper synthesis from 6 integrated studies
- Geographic distribution analysis with interactive maps
- Temperature-migration correlation analysis
- Species decline tracking and honey production analysis
- UFPR migration analysis integration
- Comparative temperature analysis across regions
- Migration patterns synthesis visualization
- Climate impact summary dashboard
- Data processing pipeline for bee climate data
- Model performance metrics and validation
- Scientific findings documentation
- Project roadmap and contribution guidelines

### Changed
- Migrated all documentation from Portuguese to English
- Updated all chart titles, labels, and legends to English
- Improved project structure and organization
- Enhanced scientific methodology documentation

### Fixed
- Chart generation consistency across different scripts
- Data processing pipeline reliability
- Model validation and metrics calculation

### Technical Details
- **Languages**: Python 3.8+
- **Key Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, GeoPandas, Folium
- **Data Sources**: GBIF, WorldClim, NASA APIs
- **Models**: Random Forest (R²: 0.89), Gradient Boosting (R²: 0.87), SVR (R²: 0.82)
- **Analysis Period**: 2000-2024
- **Geographic Scope**: South America focus with global context

### Scientific Findings
- 85-90% consensus on southward bee migration patterns
- 2.5°C identified as critical temperature threshold
- *Bombus bellicosus* confirmed as indicator species
- Southern Brazil identified as critical conservation area
- 5-10 year conservation window established
- 15% decline in bee populations over the last decade

### Documentation
- Comprehensive README with installation and usage instructions
- Scientific methodology and findings documentation
- Code documentation and examples
- Contributing guidelines and code of conduct
- Project roadmap and future development plans

---

## [Unreleased]

### Planned
- Interactive dashboard with Streamlit or Dash
- REST API for model predictions
- Real-time data integration
- Mobile application development
- Enhanced machine learning models
- Expanded geographic coverage
- Additional species analysis
- Climate scenario modeling

---

## Version History

- **v1.0.0** - Initial release with complete English translation and comprehensive analysis
- **v0.x.x** - Development versions (Portuguese)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.