#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar gráficos visuais demonstrativos sobre análise de abelhas e mudanças climáticas.
Os gráficos são salvos na pasta images/ para exibição no README.

Autor: Diego Gomes
Data: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração do estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_sample_data():
    """
    Gera dados simulados realistas para demonstração das análises.
    """
    np.random.seed(42)
    
    # Dados temporais (2000-2023)
    years = np.arange(2000, 2024)
    n_years = len(years)
    
    # Dados de temperatura (tendência de aquecimento)
    base_temp = 22.5
    temp_trend = np.linspace(0, 2.5, n_years)  # Aumento de 2.5°C em 23 anos
    temperature = base_temp + temp_trend + np.random.normal(0, 0.5, n_years)
    
    # Dados de latitude média das abelhas (migração para o sul)
    base_lat = -15.0  # Latitude inicial (sul do Brasil)
    lat_trend = np.linspace(0, -3.0, n_years)  # Migração 3° para o sul
    latitude = base_lat + lat_trend + np.random.normal(0, 0.3, n_years)
    
    # Número de espécies observadas
    base_species = 45
    species_decline = np.linspace(0, -8, n_years)  # Declínio de 8 espécies
    species_count = base_species + species_decline + np.random.normal(0, 2, n_years)
    species_count = np.maximum(species_count, 20)  # Mínimo de 20 espécies
    
    # Produtividade de mel (kg/colmeia)
    base_honey = 25.0
    honey_decline = np.linspace(0, -5, n_years)  # Declínio de 5kg
    honey_production = base_honey + honey_decline + np.random.normal(0, 1.5, n_years)
    honey_production = np.maximum(honey_production, 15)  # Mínimo de 15kg
    
    return pd.DataFrame({
        'year': years,
        'temperature': temperature,
        'latitude': latitude,
        'species_count': species_count,
        'honey_production': honey_production
    })

def generate_geographic_data():
    """
    Gera dados geográficos simulados para mapas de distribuição.
    """
    np.random.seed(42)
    
    # Coordenadas da América do Sul
    n_points = 200
    
    # Distribuição atual (mais ao norte)
    lat_current = np.random.normal(-12, 4, n_points//2)
    lon_current = np.random.normal(-55, 8, n_points//2)
    
    # Distribuição futura (mais ao sul)
    lat_future = np.random.normal(-18, 4, n_points//2)
    lon_future = np.random.normal(-58, 8, n_points//2)
    
    current_data = pd.DataFrame({
        'latitude': lat_current,
        'longitude': lon_current,
        'period': 'Atual (2020-2023)',
        'abundance': np.random.exponential(2, n_points//2)
    })
    
    future_data = pd.DataFrame({
        'latitude': lat_future,
        'longitude': lon_future,
        'period': 'Projetado (2040-2050)',
        'abundance': np.random.exponential(1.5, n_points//2)
    })
    
    return pd.concat([current_data, future_data], ignore_index=True)

def plot_temperature_migration_trend(df):
    """
    Gráfico mostrando a correlação entre temperatura e migração das abelhas.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Temperatura ao longo do tempo
    ax1.plot(df['year'], df['temperature'], 'o-', color='red', linewidth=2, markersize=6)
    ax1.set_title('Aumento da Temperatura Média Anual na América do Sul', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.grid(True, alpha=0.3)
    
    # Linha de tendência
    z = np.polyfit(df['year'], df['temperature'], 1)
    p = np.poly1d(z)
    ax1.plot(df['year'], p(df['year']), "--", color='darkred', alpha=0.8, linewidth=2)
    
    # Gráfico 2: Latitude média das abelhas
    ax2.plot(df['year'], df['latitude'], 'o-', color='blue', linewidth=2, markersize=6)
    ax2.set_title('Migração das Abelhas para o Sul (Latitude Média)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Latitude (°S)')
    ax2.grid(True, alpha=0.3)
    
    # Linha de tendência
    z2 = np.polyfit(df['year'], df['latitude'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['year'], p2(df['year']), "--", color='darkblue', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('../images/temperature_migration_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: temperature_migration_trend.png")

def plot_correlation_analysis(df):
    """
    Matriz de correlação e scatter plots.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Matriz de correlação
    corr_matrix = df[['temperature', 'latitude', 'species_count', 'honey_production']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Matriz de Correlação entre Variáveis', fontsize=14, fontweight='bold')
    
    # Scatter plot: Temperatura vs Latitude
    ax2.scatter(df['temperature'], df['latitude'], alpha=0.7, s=60, color='purple')
    ax2.set_xlabel('Temperatura (°C)')
    ax2.set_ylabel('Latitude (°S)')
    ax2.set_title('Temperatura vs Latitude das Abelhas')
    
    # Linha de regressão
    z = np.polyfit(df['temperature'], df['latitude'], 1)
    p = np.poly1d(z)
    ax2.plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8)
    
    # Scatter plot: Temperatura vs Número de Espécies
    ax3.scatter(df['temperature'], df['species_count'], alpha=0.7, s=60, color='green')
    ax3.set_xlabel('Temperatura (°C)')
    ax3.set_ylabel('Número de Espécies')
    ax3.set_title('Temperatura vs Diversidade de Espécies')
    
    # Linha de regressão
    z3 = np.polyfit(df['temperature'], df['species_count'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(df['temperature'], p3(df['temperature']), "r--", alpha=0.8)
    
    # Scatter plot: Temperatura vs Produção de Mel
    ax4.scatter(df['temperature'], df['honey_production'], alpha=0.7, s=60, color='orange')
    ax4.set_xlabel('Temperatura (°C)')
    ax4.set_ylabel('Produção de Mel (kg/colmeia)')
    ax4.set_title('Temperatura vs Produção de Mel')
    
    # Linha de regressão
    z4 = np.polyfit(df['temperature'], df['honey_production'], 1)
    p4 = np.poly1d(z4)
    ax4.plot(df['temperature'], p4(df['temperature']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('../images/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: correlation_analysis.png")

def plot_species_decline(df):
    """
    Gráfico mostrando o declínio das espécies e produção de mel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Declínio das espécies
    ax1.fill_between(df['year'], df['species_count'], alpha=0.3, color='red')
    ax1.plot(df['year'], df['species_count'], 'o-', color='darkred', linewidth=2, markersize=5)
    ax1.set_title('Declínio na Diversidade de Espécies de Abelhas', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Número de Espécies Observadas')
    ax1.grid(True, alpha=0.3)
    
    # Linha de tendência
    z = np.polyfit(df['year'], df['species_count'], 1)
    p = np.poly1d(z)
    ax1.plot(df['year'], p(df['year']), "--", color='black', alpha=0.8, linewidth=2)
    
    # Produção de mel
    ax2.fill_between(df['year'], df['honey_production'], alpha=0.3, color='gold')
    ax2.plot(df['year'], df['honey_production'], 'o-', color='darkorange', linewidth=2, markersize=5)
    ax2.set_title('Impacto na Produção de Mel', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Produção Média (kg/colmeia)')
    ax2.grid(True, alpha=0.3)
    
    # Linha de tendência
    z2 = np.polyfit(df['year'], df['honey_production'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['year'], p2(df['year']), "--", color='black', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('../images/species_honey_decline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: species_honey_decline.png")

def plot_geographic_distribution(geo_df):
    """
    Mapa mostrando a distribuição atual vs futura das abelhas.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Distribuição atual
    current = geo_df[geo_df['period'] == 'Atual (2020-2023)']
    scatter1 = ax1.scatter(current['longitude'], current['latitude'], 
                          c=current['abundance'], s=60, alpha=0.7, 
                          cmap='Reds', edgecolors='black', linewidth=0.5)
    ax1.set_title('Distribuição Atual das Abelhas\n(2020-2023)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Distribuição futura
    future = geo_df[geo_df['period'] == 'Projetado (2040-2050)']
    scatter2 = ax2.scatter(future['longitude'], future['latitude'], 
                          c=future['abundance'], s=60, alpha=0.7, 
                          cmap='Blues', edgecolors='black', linewidth=0.5)
    ax2.set_title('Distribuição Projetada das Abelhas\n(2040-2050)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    plt.colorbar(scatter1, ax=ax1, label='Abundância Relativa')
    plt.colorbar(scatter2, ax=ax2, label='Abundância Relativa')
    
    plt.tight_layout()
    plt.savefig('../images/geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: geographic_distribution.png")

def plot_climate_impact_summary(df):
    """
    Gráfico resumo mostrando todos os impactos das mudanças climáticas.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Normalizar dados para comparação (0-100)
    temp_norm = ((df['temperature'] - df['temperature'].min()) / 
                 (df['temperature'].max() - df['temperature'].min())) * 100
    
    lat_norm = ((df['latitude'].max() - df['latitude']) / 
                (df['latitude'].max() - df['latitude'].min())) * 100
    
    species_norm = ((df['species_count'] - df['species_count'].min()) / 
                    (df['species_count'].max() - df['species_count'].min())) * 100
    
    honey_norm = ((df['honey_production'] - df['honey_production'].min()) / 
                  (df['honey_production'].max() - df['honey_production'].min())) * 100
    
    # Gráfico 1: Tendências normalizadas
    ax1.plot(df['year'], temp_norm, 'o-', label='Temperatura', color='red', linewidth=2)
    ax1.plot(df['year'], lat_norm, 'o-', label='Migração Sul', color='blue', linewidth=2)
    ax1.plot(df['year'], species_norm, 'o-', label='Diversidade', color='green', linewidth=2)
    ax1.plot(df['year'], honey_norm, 'o-', label='Produção Mel', color='orange', linewidth=2)
    ax1.set_title('Tendências Normalizadas (0-100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Índice Normalizado')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Boxplot das variáveis
    data_for_box = [df['temperature'], df['latitude'].abs(), df['species_count'], df['honey_production']]
    labels_box = ['Temperatura\n(°C)', 'Latitude\n(°S)', 'Espécies\n(count)', 'Mel\n(kg)']
    
    box_plot = ax2.boxplot(data_for_box, labels=labels_box, patch_artist=True)
    colors = ['red', 'blue', 'green', 'orange']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Distribuição das Variáveis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Mudança percentual desde 2000
    base_year = df.iloc[0]
    temp_change = ((df['temperature'] - base_year['temperature']) / base_year['temperature']) * 100
    species_change = ((df['species_count'] - base_year['species_count']) / base_year['species_count']) * 100
    honey_change = ((df['honey_production'] - base_year['honey_production']) / base_year['honey_production']) * 100
    
    ax3.plot(df['year'], temp_change, 'o-', label='Temperatura', color='red', linewidth=2)
    ax3.plot(df['year'], species_change, 'o-', label='Diversidade', color='green', linewidth=2)
    ax3.plot(df['year'], honey_change, 'o-', label='Produção Mel', color='orange', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Mudança Percentual desde 2000', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Ano')
    ax3.set_ylabel('Mudança (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Projeção futura (2024-2030)
    future_years = np.arange(2024, 2031)
    
    # Extrapolação linear simples
    temp_trend = np.polyfit(df['year'], df['temperature'], 1)
    species_trend = np.polyfit(df['year'], df['species_count'], 1)
    honey_trend = np.polyfit(df['year'], df['honey_production'], 1)
    
    future_temp = np.polyval(temp_trend, future_years)
    future_species = np.polyval(species_trend, future_years)
    future_honey = np.polyval(honey_trend, future_years)
    
    # Normalizar projeções
    all_temp = np.concatenate([df['temperature'], future_temp])
    all_species = np.concatenate([df['species_count'], future_species])
    all_honey = np.concatenate([df['honey_production'], future_honey])
    
    temp_norm_proj = ((all_temp - all_temp.min()) / (all_temp.max() - all_temp.min())) * 100
    species_norm_proj = ((all_species - all_species.min()) / (all_species.max() - all_species.min())) * 100
    honey_norm_proj = ((all_honey - all_honey.min()) / (all_honey.max() - all_honey.min())) * 100
    
    all_years = np.concatenate([df['year'], future_years])
    
    ax4.plot(all_years[:len(df)], temp_norm_proj[:len(df)], 'o-', color='red', linewidth=2, label='Temperatura')
    ax4.plot(all_years[:len(df)], species_norm_proj[:len(df)], 'o-', color='green', linewidth=2, label='Diversidade')
    ax4.plot(all_years[:len(df)], honey_norm_proj[:len(df)], 'o-', color='orange', linewidth=2, label='Produção Mel')
    
    ax4.plot(all_years[len(df):], temp_norm_proj[len(df):], '--', color='red', linewidth=2, alpha=0.7)
    ax4.plot(all_years[len(df):], species_norm_proj[len(df):], '--', color='green', linewidth=2, alpha=0.7)
    ax4.plot(all_years[len(df):], honey_norm_proj[len(df):], '--', color='orange', linewidth=2, alpha=0.7)
    
    ax4.axvline(x=2023, color='black', linestyle=':', alpha=0.7, label='Projeção')
    ax4.set_title('Projeção até 2030', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Ano')
    ax4.set_ylabel('Índice Normalizado')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/climate_impact_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: climate_impact_summary.png")

def main():
    """
    Função principal que executa todas as análises e gera os gráficos.
    """
    print("Iniciando geração de gráficos para análise de abelhas...\n")
    
    # Gerar dados simulados
    print("Gerando dados simulados...")
    df = generate_sample_data()
    geo_df = generate_geographic_data()
    
    print(f"Dados gerados: {len(df)} anos de dados temporais")
    print(f"Dados geográficos: {len(geo_df)} pontos de distribuição\n")
    
    # Gerar gráficos
    print("Gerando gráficos...")
    
    plot_temperature_migration_trend(df)
    plot_correlation_analysis(df)
    plot_species_decline(df)
    plot_geographic_distribution(geo_df)
    plot_climate_impact_summary(df)
    
    print("\nTodos os gráficos foram gerados com sucesso!")
    print("Arquivos salvos na pasta 'images/'")
    print("\nGráficos gerados:")
    print("   1. temperature_migration_trend.png - Tendência de temperatura e migração")
    print("   2. correlation_analysis.png - Análise de correlações")
    print("   3. species_honey_decline.png - Declínio de espécies e produção")
    print("   4. geographic_distribution.png - Distribuição geográfica")
    print("   5. climate_impact_summary.png - Resumo dos impactos climáticos")
    
    # Salvar dados para uso posterior
    df.to_csv('../data/bee_climate_data.csv', index=False)
    geo_df.to_csv('../data/bee_geographic_data.csv', index=False)
    print("\nDados salvos em 'data/' para análises futuras")

if __name__ == "__main__":
    main()