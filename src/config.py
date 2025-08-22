#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuração do projeto de análise de abelhas e mudanças climáticas.
Centraliza todas as configurações, caminhos e parâmetros do projeto.

Autor: Diego Gomes
Data: 2024
"""

import os
import logging
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
DOCS_DIR = BASE_DIR / "docs"

# Subdiretórios de dados
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                  RESULTS_DIR, IMAGES_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurações de dados
DATA_CONFIG = {
    # URLs de APIs climáticas
    'CLIMATE_APIS': {
        'WORLDBANK': 'https://climateknowledgeportal.worldbank.org/api/data',
        'NOAA': 'https://www.ncdc.noaa.gov/cdo-web/api/v2',
        'ECMWF': 'https://cds.climate.copernicus.eu/api/v2'
    },
    
    # URLs de dados de biodiversidade
    'BIODIVERSITY_APIS': {
        'GBIF': 'https://api.gbif.org/v1',
        'IUCN': 'https://apiv3.iucnredlist.org/api/v3',
        'EOL': 'https://eol.org/api'
    },
    
    # Configurações de amostragem
    'SAMPLING': {
        'START_YEAR': 2000,
        'END_YEAR': 2023,
        'GEOGRAPHIC_BOUNDS': {
            'NORTH': 12.0,   # Venezuela
            'SOUTH': -55.0,  # Argentina
            'WEST': -81.0,   # Peru
            'EAST': -34.0    # Brasil
        }
    },
    
    # Espécies de abelhas para análise
    'TARGET_SPECIES': [
        'Apis mellifera',
        'Melipona quadrifasciata',
        'Trigona spinipes',
        'Scaptotrigona bipunctata',
        'Nannotrigona testaceicornis'
    ]
}

# Configurações de análise
ANALYSIS_CONFIG = {
    # Parâmetros de análise temporal
    'TEMPORAL': {
        'AGGREGATION_PERIODS': ['monthly', 'seasonal', 'yearly'],
        'TREND_ANALYSIS_WINDOW': 5,  # anos
        'SEASONAL_DECOMPOSITION': True
    },
    
    # Parâmetros de análise espacial
    'SPATIAL': {
        'GRID_RESOLUTION': 0.5,  # graus
        'INTERPOLATION_METHOD': 'kriging',
        'BUFFER_DISTANCE': 50  # km
    },
    
    # Parâmetros de correlação
    'CORRELATION': {
        'MIN_CORRELATION_THRESHOLD': 0.3,
        'SIGNIFICANCE_LEVEL': 0.05,
        'LAG_ANALYSIS_PERIODS': [0, 1, 2, 3, 6, 12]  # meses
    }
}

# Configurações de Machine Learning
ML_CONFIG = {
    # Divisão dos dados
    'DATA_SPLIT': {
        'TRAIN_SIZE': 0.7,
        'VALIDATION_SIZE': 0.15,
        'TEST_SIZE': 0.15,
        'RANDOM_STATE': 42
    },
    
    # Modelos a serem testados
    'MODELS': {
        'LINEAR_REGRESSION': {
            'enabled': True,
            'params': {}
        },
        'RANDOM_FOREST': {
            'enabled': True,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GRADIENT_BOOSTING': {
            'enabled': True,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SUPPORT_VECTOR_REGRESSION': {
            'enabled': True,
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    },
    
    # Métricas de avaliação
    'METRICS': ['rmse', 'mae', 'r2', 'mape'],
    
    # Validação cruzada
    'CROSS_VALIDATION': {
        'cv_folds': 5,
        'scoring': 'neg_mean_squared_error'
    }
}

# Configurações de visualização
VISUALIZATION_CONFIG = {
    # Configurações gerais de plots
    'PLOT_STYLE': 'seaborn-v0_8',
    'FIGURE_SIZE': (12, 8),
    'DPI': 300,
    'COLOR_PALETTE': 'husl',
    
    # Configurações específicas por tipo de gráfico
    'CHART_CONFIGS': {
        'correlation_heatmap': {
            'cmap': 'RdYlBu_r',
            'center': 0,
            'annot': True
        },
        'time_series': {
            'linewidth': 2,
            'marker': 'o',
            'markersize': 6
        },
        'scatter_plot': {
            'alpha': 0.7,
            's': 60
        },
        'geographic_map': {
            'projection': 'PlateCarree',
            'resolution': '50m'
        }
    },
    
    # Configurações de salvamento
    'SAVE_CONFIG': {
        'format': 'png',
        'bbox_inches': 'tight',
        'transparent': False
    }
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': str(RESULTS_DIR / 'analysis.log')
}

# Configurações de performance
PERFORMANCE_CONFIG = {
    'PARALLEL_PROCESSING': {
        'enabled': True,
        'n_jobs': -1,  # usar todos os cores disponíveis
        'backend': 'loky'
    },
    
    'MEMORY_OPTIMIZATION': {
        'chunk_size': 10000,
        'low_memory': True
    }
}

# Configurações específicas do projeto
PROJECT_CONFIG = {
    'NAME': 'Bee Climate Analysis',
    'VERSION': '1.0.0',
    'AUTHOR': 'Diego Gomes',
    'DESCRIPTION': 'Análise do impacto das mudanças climáticas no comportamento de abelhas na América do Sul',
    'GITHUB_URL': 'https://github.com/digomes87/Bees',
    
    # Configurações de relatório
    'REPORT': {
        'TITLE': 'Impacto das Mudanças Climáticas nas Abelhas Sul-Americanas',
        'SUBTITLE': 'Análise Temporal e Espacial com Machine Learning',
        'LANGUAGE': 'pt-BR',
        'TEMPLATE': 'professional'
    }
}

# Função para validar configurações
def validate_config():
    """
    Valida se todas as configurações estão corretas.
    """
    errors = []
    
    # Verificar se diretórios existem
    required_dirs = [DATA_DIR, RESULTS_DIR, IMAGES_DIR]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Diretório não encontrado: {directory}")
    
    # Verificar configurações de ML
    total_split = (ML_CONFIG['DATA_SPLIT']['TRAIN_SIZE'] + 
                   ML_CONFIG['DATA_SPLIT']['VALIDATION_SIZE'] + 
                   ML_CONFIG['DATA_SPLIT']['TEST_SIZE'])
    
    if abs(total_split - 1.0) > 0.001:
        errors.append("Soma das proporções de divisão dos dados deve ser 1.0")
    
    # Verificar bounds geográficos
    bounds = DATA_CONFIG['SAMPLING']['GEOGRAPHIC_BOUNDS']
    if bounds['NORTH'] <= bounds['SOUTH'] or bounds['EAST'] <= bounds['WEST']:
        errors.append("Bounds geográficos inválidos")
    
    if errors:
        raise ValueError(f"Erros de configuração encontrados: {'; '.join(errors)}")
    
    return True

# Função para obter configuração específica
def get_config(section, key=None):
    """
    Obtém configuração específica.
    
    Args:
        section (str): Seção da configuração
        key (str, optional): Chave específica dentro da seção
    
    Returns:
        dict or any: Configuração solicitada
    """
    config_map = {
        'data': DATA_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'ml': ML_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'project': PROJECT_CONFIG
    }
    
    if section not in config_map:
        raise ValueError(f"Seção de configuração '{section}' não encontrada")
    
    config = config_map[section]
    
    if key is None:
        return config
    
    if key not in config:
        raise ValueError(f"Chave '{key}' não encontrada na seção '{section}'")
    
    return config[key]

if __name__ == "__main__":
    # Validar configurações ao importar
    validate_config()
    print("✅ Configurações validadas com sucesso!")
    print(f"📁 Diretório base: {BASE_DIR}")
    print(f"📊 Projeto: {PROJECT_CONFIG['NAME']} v{PROJECT_CONFIG['VERSION']}")