#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preditor de Migração de Abelhas usando Machine Learning.

Este módulo implementa modelos de machine learning para prever padrões de migração
de abelhas baseado em variáveis climáticas e ambientais.

Autor: Diego Gomes
Data: 2024
Projeto: Bee Climate Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Imports locais
from config import (
    ML_CONFIG, VISUALIZATION_CONFIG, RESULTS_DIR, 
    IMAGES_DIR, PROCESSED_DATA_DIR
)
from bee_analysis import BeeClimateAnalyzer

# Configurar warnings e logging
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BeeMigrationPredictor:
    """
    Classe para predição de migração de abelhas usando machine learning.
    """
    
    def __init__(self):
        """
        Inicializa o preditor com configurações padrão.
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        self.predictions = {}
        self.metrics = {}
        
        logger.info("BeeMigrationPredictor inicializado")
    
    def load_and_prepare_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega e prepara dados para machine learning.
        
        Args:
            data_path (str, optional): Caminho para arquivo de dados
        
        Returns:
            pd.DataFrame: Dados preparados
        """
        logger.info("Carregando e preparando dados para ML...")
        
        if data_path and Path(data_path).exists():
            self.data = pd.read_csv(data_path)
        else:
            # Usar dados do analisador principal
            analyzer = BeeClimateAnalyzer()
            self.data = analyzer.load_data()
        
        # Preparar features e targets
        self.data = self._engineer_features(self.data)
        
        logger.info(f"Dados preparados: {len(self.data)} registros, {self.data.shape[1]} features")
        return self.data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engenharia de features para machine learning.
        
        Args:
            data (pd.DataFrame): Dados brutos
        
        Returns:
            pd.DataFrame: Dados com features engineered
        """
        logger.info("Realizando engenharia de features...")
        
        # Criar features temporais
        data['year_normalized'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
        data['decade'] = (data['year'] // 10) * 10
        
        # Features climáticas derivadas
        data['temp_precip_ratio'] = data['temperature'] / (data['precipitation'] + 1)
        data['climate_extremity'] = np.sqrt(data['temperature_anomaly']**2 + data['precipitation_anomaly']**2)
        
        # Features de interação
        data['temp_lat_interaction'] = data['temperature'] * data['latitude']
        data['precip_lat_interaction'] = data['precipitation'] * data['latitude']
        
        # Features de lag (valores do ano anterior)
        data_sorted = data.sort_values(['region', 'year'])
        for col in ['temperature', 'precipitation', 'bee_abundance']:
            data_sorted[f'{col}_lag1'] = data_sorted.groupby('region')[col].shift(1)
            data_sorted[f'{col}_change'] = data_sorted[col] - data_sorted[f'{col}_lag1']
        
        # Features estatísticas por região
        for col in ['temperature', 'precipitation']:
            data_sorted[f'{col}_region_mean'] = data_sorted.groupby('region')[col].transform('mean')
            data_sorted[f'{col}_region_std'] = data_sorted.groupby('region')[col].transform('std')
            data_sorted[f'{col}_zscore'] = (data_sorted[col] - data_sorted[f'{col}_region_mean']) / data_sorted[f'{col}_region_std']
        
        # Encoding de variáveis categóricas
        le_region = LabelEncoder()
        data_sorted['region_encoded'] = le_region.fit_transform(data_sorted['region'])
        
        # Salvar encoder para uso posterior
        joblib.dump(le_region, RESULTS_DIR / 'region_encoder.pkl')
        
        return data_sorted.fillna(method='bfill').fillna(method='ffill')
    
    def prepare_ml_data(self, target_variable: str = 'bee_abundance') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados para machine learning.
        
        Args:
            target_variable (str): Variável target para predição
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) e target (y)
        """
        logger.info(f"Preparando dados ML para target: {target_variable}")
        
        # Selecionar features
        feature_columns = [
            'year_normalized', 'latitude', 'longitude', 'temperature', 'precipitation',
            'temperature_anomaly', 'precipitation_anomaly', 'climate_stress_index',
            'temp_precip_ratio', 'climate_extremity', 'temp_lat_interaction',
            'precip_lat_interaction', 'region_encoded'
        ]
        
        # Adicionar features de lag se disponíveis
        lag_features = [col for col in self.data.columns if 'lag1' in col or 'change' in col or 'zscore' in col]
        feature_columns.extend(lag_features)
        
        # Remover features que não existem
        feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        X = self.data[feature_columns].values
        y = self.data[target_variable].values
        
        # Remover linhas com NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Dados ML preparados: {X.shape[0]} amostras, {X.shape[1]} features")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Divide dados em treino, validação e teste.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
        """
        config = ML_CONFIG['DATA_SPLIT']
        
        # Primeira divisão: treino+validação vs teste
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, 
            test_size=config['TEST_SIZE'],
            random_state=config['RANDOM_STATE'],
            stratify=None  # Para regressão
        )
        
        # Segunda divisão: treino vs validação
        val_size = config['VALIDATION_SIZE'] / (config['TRAIN_SIZE'] + config['VALIDATION_SIZE'])
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=config['RANDOM_STATE']
        )
        
        # Normalizar features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Dados divididos - Treino: {len(self.X_train)}, Validação: {len(self.X_val)}, Teste: {len(self.X_test)}")
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Inicializa modelos de machine learning.
        
        Returns:
            Dict[str, Any]: Dicionário com modelos inicializados
        """
        models = {}
        
        model_configs = ML_CONFIG['MODELS']
        
        if model_configs['LINEAR_REGRESSION']['enabled']:
            models['Linear Regression'] = LinearRegression()
        
        if model_configs['RANDOM_FOREST']['enabled']:
            models['Random Forest'] = RandomForestRegressor(
                random_state=ML_CONFIG['DATA_SPLIT']['RANDOM_STATE'],
                n_jobs=-1
            )
        
        if model_configs['GRADIENT_BOOSTING']['enabled']:
            models['Gradient Boosting'] = GradientBoostingRegressor(
                random_state=ML_CONFIG['DATA_SPLIT']['RANDOM_STATE']
            )
        
        if model_configs['SUPPORT_VECTOR_REGRESSION']['enabled']:
            models['SVR'] = SVR()
        
        self.models = models
        logger.info(f"Modelos inicializados: {list(models.keys())}")
        
        return models
    
    def train_models(self) -> Dict[str, Any]:
        """
        Treina todos os modelos com otimização de hiperparâmetros.
        
        Returns:
            Dict[str, Any]: Modelos treinados
        """
        logger.info("Iniciando treinamento dos modelos...")
        
        model_configs = ML_CONFIG['MODELS']
        cv_config = ML_CONFIG['CROSS_VALIDATION']
        
        for name, model in self.models.items():
            logger.info(f"Treinando {name}...")
            
            # Obter parâmetros para grid search
            if name == 'Random Forest' and model_configs['RANDOM_FOREST']['params']:
                param_grid = model_configs['RANDOM_FOREST']['params']
            elif name == 'Gradient Boosting' and model_configs['GRADIENT_BOOSTING']['params']:
                param_grid = model_configs['GRADIENT_BOOSTING']['params']
            elif name == 'SVR' and model_configs['SUPPORT_VECTOR_REGRESSION']['params']:
                param_grid = model_configs['SUPPORT_VECTOR_REGRESSION']['params']
            else:
                param_grid = {}
            
            if param_grid:
                # Grid search com validação cruzada
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_config['cv_folds'],
                    scoring=cv_config['scoring'],
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(self.X_train, self.y_train)
                self.best_models[name] = grid_search.best_estimator_
                
                logger.info(f"{name} - Melhores parâmetros: {grid_search.best_params_}")
            else:
                # Treinar modelo simples
                model.fit(self.X_train, self.y_train)
                self.best_models[name] = model
        
        logger.info("Treinamento concluído")
        return self.best_models
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Avalia performance dos modelos.
        
        Returns:
            Dict[str, Dict[str, float]]: Métricas de cada modelo
        """
        logger.info("Avaliando performance dos modelos...")
        
        metrics = {}
        
        for name, model in self.best_models.items():
            # Predições
            y_pred_train = model.predict(self.X_train)
            y_pred_val = model.predict(self.X_val)
            y_pred_test = model.predict(self.X_test)
            
            # Salvar predições
            self.predictions[name] = {
                'train': y_pred_train,
                'validation': y_pred_val,
                'test': y_pred_test
            }
            
            # Calcular métricas
            model_metrics = {}
            
            for dataset, y_true, y_pred in [
                ('train', self.y_train, y_pred_train),
                ('validation', self.y_val, y_pred_val),
                ('test', self.y_test, y_pred_test)
            ]:
                model_metrics[f'{dataset}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                model_metrics[f'{dataset}_mae'] = mean_absolute_error(y_true, y_pred)
                model_metrics[f'{dataset}_r2'] = r2_score(y_true, y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                model_metrics[f'{dataset}_mape'] = mape
            
            metrics[name] = model_metrics
            
            logger.info(f"{name} - Test R²: {model_metrics['test_r2']:.3f}, Test RMSE: {model_metrics['test_rmse']:.3f}")
        
        self.metrics = metrics
        return metrics
    
    def analyze_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Analisa importância das features.
        
        Returns:
            Dict[str, np.ndarray]: Importância das features por modelo
        """
        logger.info("Analisando importância das features...")
        
        feature_names = [
            'year_normalized', 'latitude', 'longitude', 'temperature', 'precipitation',
            'temperature_anomaly', 'precipitation_anomaly', 'climate_stress_index',
            'temp_precip_ratio', 'climate_extremity', 'temp_lat_interaction',
            'precip_lat_interaction', 'region_encoded'
        ]
        
        # Adicionar nomes de features de lag
        lag_features = [col for col in self.data.columns if 'lag1' in col or 'change' in col or 'zscore' in col]
        feature_names.extend(lag_features)
        
        # Ajustar para o número real de features
        feature_names = feature_names[:self.X_train.shape[1]]
        
        importance = {}
        
        for name, model in self.best_models.items():
            if hasattr(model, 'feature_importances_'):
                # Modelos tree-based
                importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Modelos lineares
                importance[name] = np.abs(model.coef_)
            else:
                # Para outros modelos, usar permutation importance
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(
                    model, self.X_val, self.y_val, n_repeats=10, random_state=42
                )
                importance[name] = perm_importance.importances_mean
        
        self.feature_importance = importance
        
        # Criar DataFrame para facilitar análise
        importance_df = pd.DataFrame(importance, index=feature_names)
        importance_df.to_csv(RESULTS_DIR / 'feature_importance.csv')
        
        logger.info("Análise de importância concluída")
        return importance
    
    def generate_predictions_report(self) -> str:
        """
        Gera relatório detalhado das predições.
        
        Returns:
            str: Relatório em formato texto
        """
        logger.info("Gerando relatório de predições...")
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE PREDIÇÕES - MACHINE LEARNING")
        report.append("=" * 80)
        report.append(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
        report.append(f"Modelos treinados: {len(self.best_models)}")
        report.append(f"Features utilizadas: {self.X_train.shape[1]}")
        report.append("")
        
        # Performance dos modelos
        report.append("PERFORMANCE DOS MODELOS")
        report.append("-" * 40)
        
        # Encontrar melhor modelo
        best_model_name = max(self.metrics.keys(), 
                             key=lambda x: self.metrics[x]['test_r2'])
        
        for name, metrics in self.metrics.items():
            marker = " ⭐" if name == best_model_name else ""
            report.append(f"{name}{marker}:")
            report.append(f"  Test R²: {metrics['test_r2']:.4f}")
            report.append(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            report.append(f"  Test MAE: {metrics['test_mae']:.4f}")
            report.append(f"  Test MAPE: {metrics['test_mape']:.2f}%")
            report.append("")
        
        # Features mais importantes
        if self.feature_importance:
            report.append("FEATURES MAIS IMPORTANTES")
            report.append("-" * 40)
            
            # Usar o melhor modelo para mostrar importância
            if best_model_name in self.feature_importance:
                importance = self.feature_importance[best_model_name]
                feature_names = [
                    'year_normalized', 'latitude', 'longitude', 'temperature', 'precipitation',
                    'temperature_anomaly', 'precipitation_anomaly', 'climate_stress_index',
                    'temp_precip_ratio', 'climate_extremity', 'temp_lat_interaction',
                    'precip_lat_interaction', 'region_encoded'
                ]
                
                # Ajustar para o número real de features
                feature_names = feature_names[:len(importance)]
                
                # Ordenar por importância
                sorted_indices = np.argsort(importance)[::-1]
                
                for i, idx in enumerate(sorted_indices[:10]):  # Top 10
                    if idx < len(feature_names):
                        report.append(f"{i+1:2d}. {feature_names[idx]}: {importance[idx]:.4f}")
            
            report.append("")
        
        # Recomendações
        report.append("RECOMENDAÇÕES")
        report.append("-" * 40)
        report.append(f"1. Melhor modelo: {best_model_name}")
        report.append(f"2. R² de {self.metrics[best_model_name]['test_r2']:.3f} indica {'boa' if self.metrics[best_model_name]['test_r2'] > 0.7 else 'moderada'} capacidade preditiva")
        report.append("3. Considerar coleta de mais dados para melhorar performance")
        report.append("4. Monitorar predições em dados futuros para validação")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Salvar relatório
        report_path = RESULTS_DIR / "ml_predictions_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Relatório de predições salvo em: {report_path}")
        return report_text
    
    def save_models(self):
        """
        Salva modelos treinados.
        """
        logger.info("Salvando modelos treinados...")
        
        models_dir = RESULTS_DIR / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.best_models.items():
            model_path = models_dir / f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
        
        # Salvar scaler
        joblib.dump(self.scaler, models_dir / "scaler.pkl")
        
        # Salvar métricas
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv")
        
        logger.info(f"Modelos salvos em: {models_dir}")
    
    def run_complete_ml_pipeline(self, data_path: Optional[str] = None, 
                                target_variable: str = 'bee_abundance') -> Dict:
        """
        Executa pipeline completo de machine learning.
        
        Args:
            data_path (str, optional): Caminho para dados
            target_variable (str): Variável target
        
        Returns:
            Dict: Resultados completos do ML
        """
        logger.info("Iniciando pipeline completo de ML...")
        
        # Carregar e preparar dados
        self.load_and_prepare_data(data_path)
        
        # Preparar dados para ML
        X, y = self.prepare_ml_data(target_variable)
        
        # Dividir dados
        self.split_data(X, y)
        
        # Inicializar e treinar modelos
        self.initialize_models()
        self.train_models()
        
        # Avaliar modelos
        metrics = self.evaluate_models()
        
        # Analisar importância das features
        feature_importance = self.analyze_feature_importance()
        
        # Gerar relatório
        report = self.generate_predictions_report()
        
        # Salvar modelos
        self.save_models()
        
        logger.info("Pipeline de ML concluído")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': self.predictions,
            'best_models': self.best_models,
            'report': report
        }

def main():
    """
    Função principal para executar predições ML.
    """
    print("🤖 Iniciando Predições de Migração de Abelhas com ML")
    print("=" * 60)
    
    # Criar preditor
    predictor = BeeMigrationPredictor()
    
    # Executar pipeline completo
    results = predictor.run_complete_ml_pipeline()
    
    # Exibir resumo
    print("\n📊 RESUMO DOS RESULTADOS")
    print("-" * 30)
    
    best_model = max(results['metrics'].keys(), 
                    key=lambda x: results['metrics'][x]['test_r2'])
    
    print(f"Melhor modelo: {best_model}")
    print(f"R² no teste: {results['metrics'][best_model]['test_r2']:.4f}")
    print(f"RMSE no teste: {results['metrics'][best_model]['test_rmse']:.4f}")
    print(f"MAE no teste: {results['metrics'][best_model]['test_mae']:.4f}")
    
    print("\n✅ Pipeline de ML concluído com sucesso!")
    print(f"📁 Modelos salvos em: {RESULTS_DIR / 'models'}")
    print(f"📊 Métricas salvas em: {RESULTS_DIR}")
    
    return results

if __name__ == "__main__":
    results = main()