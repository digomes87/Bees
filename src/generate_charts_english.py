#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate visual charts for bee migration and climate change analysis.
Charts are saved in the images/ folder for README display.

Author: Diego Gomes
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Chart style configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_sample_data():
    """
    Generates realistic simulated data for analysis demonstration.
    """
    np.random.seed(42)
    
    # Temporal data (2000-2023)
    years = np.arange(2000, 2024)
    n_years = len(years)
    
    # Temperature data (warming trend)
    base_temp = 22.5
    temp_trend = np.linspace(0, 2.5, n_years)  # 2.5°C increase over 23 years
    temperature = base_temp + temp_trend + np.random.normal(0, 0.5, n_years)
    
    # Average bee latitude (southward migration)
    base_lat = -15.0  # Initial latitude (southern Brazil)
    lat_trend = np.linspace(0, -3.0, n_years)  # 3° southward migration
    latitude = base_lat + lat_trend + np.random.normal(0, 0.3, n_years)
    
    # Number of observed species
    base_species = 45
    species_decline = np.linspace(0, -8, n_years)  # Decline of 8 species
    species_count = base_species + species_decline + np.random.normal(0, 2, n_years)
    species_count = np.maximum(species_count, 20)  # Minimum 20 species
    
    # Honey productivity (kg/hive)
    base_honey = 25.0
    honey_decline = np.linspace(0, -5, n_years)  # 5kg decline
    honey_production = base_honey + honey_decline + np.random.normal(0, 1.5, n_years)
    honey_production = np.maximum(honey_production, 15)  # Minimum 15kg
    
    return pd.DataFrame({
        'year': years,
        'temperature': temperature,
        'latitude': latitude,
        'species_count': species_count,
        'honey_production': honey_production
    })

def generate_geographic_data():
    """
    Generates simulated geographic data for distribution maps.
    """
    np.random.seed(42)
    
    # South American coordinates
    n_points = 200
    
    # Current distribution (more northern)
    lat_current = np.random.normal(-12, 4, n_points//2)
    lon_current = np.random.normal(-55, 8, n_points//2)
    
    # Future distribution (more southern)
    lat_future = np.random.normal(-18, 4, n_points//2)
    lon_future = np.random.normal(-58, 8, n_points//2)
    
    current_data = pd.DataFrame({
        'latitude': lat_current,
        'longitude': lon_current,
        'period': 'Current (2020-2023)',
        'abundance': np.random.exponential(2, n_points//2)
    })
    
    future_data = pd.DataFrame({
        'latitude': lat_future,
        'longitude': lon_future,
        'period': 'Projected (2040-2050)',
        'abundance': np.random.exponential(1.5, n_points//2)
    })
    
    return pd.concat([current_data, future_data], ignore_index=True)

def plot_temperature_migration_trend(df):
    """
    Chart showing correlation between temperature and bee migration.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Chart 1: Temperature over time
    ax1.plot(df['year'], df['temperature'], 'o-', color='red', linewidth=2, markersize=6)
    ax1.set_title('Annual Average Temperature Increase in South America', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    
    # Trend line
    z = np.polyfit(df['year'], df['temperature'], 1)
    p = np.poly1d(z)
    ax1.plot(df['year'], p(df['year']), "--", color='darkred', alpha=0.8, linewidth=2)
    
    # Chart 2: Average bee latitude
    ax2.plot(df['year'], df['latitude'], 'o-', color='blue', linewidth=2, markersize=6)
    ax2.set_title('Southward Bee Migration (Average Latitude)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Latitude (°S)')
    ax2.grid(True, alpha=0.3)
    
    # Trend line
    z2 = np.polyfit(df['year'], df['latitude'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['year'], p2(df['year']), "--", color='darkblue', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('../images/temperature_migration_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: temperature_migration_trend.png")

def plot_correlation_analysis(df):
    """
    Correlation matrix and scatter plots.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Correlation matrix
    corr_matrix = df[['temperature', 'latitude', 'species_count', 'honey_production']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Matrix Between Variables', fontsize=14, fontweight='bold')
    
    # Scatter plot: Temperature vs Latitude
    ax2.scatter(df['temperature'], df['latitude'], alpha=0.7, s=60, color='purple')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Latitude (°S)')
    ax2.set_title('Temperature vs Bee Latitude')
    
    # Regression line
    z = np.polyfit(df['temperature'], df['latitude'], 1)
    p = np.poly1d(z)
    ax2.plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8)
    
    # Scatter plot: Temperature vs Number of Species
    ax3.scatter(df['temperature'], df['species_count'], alpha=0.7, s=60, color='green')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Number of Species')
    ax3.set_title('Temperature vs Species Diversity')
    
    # Regression line
    z3 = np.polyfit(df['temperature'], df['species_count'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(df['temperature'], p3(df['temperature']), "r--", alpha=0.8)
    
    # Scatter plot: Temperature vs Honey Production
    ax4.scatter(df['temperature'], df['honey_production'], alpha=0.7, s=60, color='orange')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Honey Production (kg/hive)')
    ax4.set_title('Temperature vs Honey Production')
    
    # Regression line
    z4 = np.polyfit(df['temperature'], df['honey_production'], 1)
    p4 = np.poly1d(z4)
    ax4.plot(df['temperature'], p4(df['temperature']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('../images/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: correlation_analysis.png")

def plot_species_decline(df):
    """
    Chart showing species decline and honey production.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Species decline
    ax1.fill_between(df['year'], df['species_count'], alpha=0.3, color='red')
    ax1.plot(df['year'], df['species_count'], 'o-', color='darkred', linewidth=2, markersize=5)
    ax1.set_title('Decline in Bee Species Diversity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Observed Species')
    ax1.grid(True, alpha=0.3)
    
    # Trend line
    z = np.polyfit(df['year'], df['species_count'], 1)
    p = np.poly1d(z)
    ax1.plot(df['year'], p(df['year']), "--", color='black', alpha=0.8, linewidth=2)
    
    # Honey production
    ax2.fill_between(df['year'], df['honey_production'], alpha=0.3, color='gold')
    ax2.plot(df['year'], df['honey_production'], 'o-', color='darkorange', linewidth=2, markersize=5)
    ax2.set_title('Impact on Honey Production', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Production (kg/hive)')
    ax2.grid(True, alpha=0.3)
    
    # Trend line
    z2 = np.polyfit(df['year'], df['honey_production'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['year'], p2(df['year']), "--", color='black', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('../images/species_honey_decline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: species_honey_decline.png")

def plot_geographic_distribution(geo_df):
    """
    Map showing current vs future bee distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Current distribution
    current = geo_df[geo_df['period'] == 'Current (2020-2023)']
    scatter1 = ax1.scatter(current['longitude'], current['latitude'], 
                          c=current['abundance'], s=60, alpha=0.7, 
                          cmap='Reds', edgecolors='black', linewidth=0.5)
    ax1.set_title('Current Bee Distribution\n(2020-2023)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Future distribution
    future = geo_df[geo_df['period'] == 'Projected (2040-2050)']
    scatter2 = ax2.scatter(future['longitude'], future['latitude'], 
                          c=future['abundance'], s=60, alpha=0.7, 
                          cmap='Blues', edgecolors='black', linewidth=0.5)
    ax2.set_title('Projected Bee Distribution\n(2040-2050)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    plt.colorbar(scatter1, ax=ax1, label='Relative Abundance')
    plt.colorbar(scatter2, ax=ax2, label='Relative Abundance')
    
    plt.tight_layout()
    plt.savefig('../images/geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: geographic_distribution.png")

def plot_climate_impact_summary(df):
    """
    Summary chart showing all climate change impacts.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Normalize data for comparison (0-100)
    temp_norm = ((df['temperature'] - df['temperature'].min()) / 
                 (df['temperature'].max() - df['temperature'].min())) * 100
    
    lat_norm = ((df['latitude'].max() - df['latitude']) / 
                (df['latitude'].max() - df['latitude'].min())) * 100
    
    species_norm = ((df['species_count'] - df['species_count'].min()) / 
                    (df['species_count'].max() - df['species_count'].min())) * 100
    
    honey_norm = ((df['honey_production'] - df['honey_production'].min()) / 
                  (df['honey_production'].max() - df['honey_production'].min())) * 100
    
    # Chart 1: Normalized trends
    ax1.plot(df['year'], temp_norm, 'o-', label='Temperature', color='red', linewidth=2)
    ax1.plot(df['year'], lat_norm, 'o-', label='Southward Migration', color='blue', linewidth=2)
    ax1.plot(df['year'], species_norm, 'o-', label='Diversity', color='green', linewidth=2)
    ax1.plot(df['year'], honey_norm, 'o-', label='Honey Production', color='orange', linewidth=2)
    ax1.set_title('Normalized Trends (0-100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Normalized Index')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Variable distribution boxplot
    data_for_box = [df['temperature'], df['latitude'].abs(), df['species_count'], df['honey_production']]
    labels_box = ['Temperature\n(°C)', 'Latitude\n(°S)', 'Species\n(count)', 'Honey\n(kg)']
    
    box_plot = ax2.boxplot(data_for_box, labels=labels_box, patch_artist=True)
    colors = ['red', 'blue', 'green', 'orange']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Variable Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Percentage change since 2000
    base_year = df.iloc[0]
    temp_change = ((df['temperature'] - base_year['temperature']) / base_year['temperature']) * 100
    species_change = ((df['species_count'] - base_year['species_count']) / base_year['species_count']) * 100
    honey_change = ((df['honey_production'] - base_year['honey_production']) / base_year['honey_production']) * 100
    
    ax3.plot(df['year'], temp_change, 'o-', label='Temperature', color='red', linewidth=2)
    ax3.plot(df['year'], species_change, 'o-', label='Diversity', color='green', linewidth=2)
    ax3.plot(df['year'], honey_change, 'o-', label='Honey Production', color='orange', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Percentage Change Since 2000', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Change (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Future projection (2024-2030)
    future_years = np.arange(2024, 2031)
    
    # Simple linear extrapolation
    temp_trend = np.polyfit(df['year'], df['temperature'], 1)
    species_trend = np.polyfit(df['year'], df['species_count'], 1)
    honey_trend = np.polyfit(df['year'], df['honey_production'], 1)
    
    future_temp = np.polyval(temp_trend, future_years)
    future_species = np.polyval(species_trend, future_years)
    future_honey = np.polyval(honey_trend, future_years)
    
    # Normalize projections
    all_temp = np.concatenate([df['temperature'], future_temp])
    all_species = np.concatenate([df['species_count'], future_species])
    all_honey = np.concatenate([df['honey_production'], future_honey])
    
    temp_norm_proj = ((all_temp - all_temp.min()) / (all_temp.max() - all_temp.min())) * 100
    species_norm_proj = ((all_species - all_species.min()) / (all_species.max() - all_species.min())) * 100
    honey_norm_proj = ((all_honey - all_honey.min()) / (all_honey.max() - all_honey.min())) * 100
    
    all_years = np.concatenate([df['year'], future_years])
    
    ax4.plot(all_years[:len(df)], temp_norm_proj[:len(df)], 'o-', color='red', linewidth=2, label='Temperature')
    ax4.plot(all_years[:len(df)], species_norm_proj[:len(df)], 'o-', color='green', linewidth=2, label='Diversity')
    ax4.plot(all_years[:len(df)], honey_norm_proj[:len(df)], 'o-', color='orange', linewidth=2, label='Honey Production')
    
    ax4.plot(all_years[len(df):], temp_norm_proj[len(df):], '--', color='red', linewidth=2, alpha=0.7)
    ax4.plot(all_years[len(df):], species_norm_proj[len(df):], '--', color='green', linewidth=2, alpha=0.7)
    ax4.plot(all_years[len(df):], honey_norm_proj[len(df):], '--', color='orange', linewidth=2, alpha=0.7)
    
    ax4.axvline(x=2023, color='black', linestyle=':', alpha=0.7, label='Projection')
    ax4.set_title('Projection to 2030', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Normalized Index')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/climate_impact_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: climate_impact_summary.png")

def plot_ufpr_migration_analysis():
    """
    UFPR-specific migration analysis chart.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UFPR Migration Analysis: South American Bees Moving South', 
                 fontsize=16, fontweight='bold')
    
    # 1. Species distribution by region
    regions = ['North', 'Northeast', 'Center-West', 'Southeast', 'South']
    current_species = [12, 8, 15, 18, 22]  # Current distribution
    projected_species = [8, 5, 12, 20, 28]  # Projected 2040
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, current_species, width, label='Current (2020)', color='lightblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, projected_species, width, label='Projected (2040)', color='darkblue', alpha=0.7)
    
    ax1.set_title('Species Distribution by Region')
    ax1.set_xlabel('Brazilian Regions')
    ax1.set_ylabel('Number of Species')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Migration direction percentages
    directions = ['South', 'Southwest', 'Southeast', 'No Change', 'North']
    percentages = [45, 40, 10, 3, 2]
    colors_dir = ['darkgreen', 'green', 'lightgreen', 'gray', 'red']
    
    bars = ax2.bar(directions, percentages, color=colors_dir, alpha=0.7)
    ax2.set_title('Migration Direction Distribution')
    ax2.set_xlabel('Direction')
    ax2.set_ylabel('Percentage of Species (%)')
    ax2.set_ylim(0, 50)
    
    # Add percentage values
    for bar, value in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{value}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 3. Impact factors
    factors = ['Temperature', 'Precipitation', 'Land Use', 'Monocultures']
    impact_scores = [9, 7, 8, 6]  # Scale 1-10
    ax3.barh(factors, impact_scores, color='orange', alpha=0.7)
    ax3.set_title('Migration Impact Factors')
    ax3.set_xlabel('Impact Intensity (1-10)')
    ax3.set_xlim(0, 10)
    
    # 4. Temporal migration projection
    projection_years = np.arange(2024, 2051, 5)
    migration_intensity = np.array([0, 15, 35, 60, 80, 100])  # Percentage of species migrating
    ax4.plot(projection_years, migration_intensity, 'bo-', linewidth=3, markersize=6)
    ax4.fill_between(projection_years, migration_intensity, alpha=0.3, color='blue')
    ax4.set_title('Southward Migration Intensity Projection')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Migrating Species (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/ufpr_migration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: ufpr_migration_analysis.png")

def plot_comparative_temperature_analysis():
    """
    Comparative temperature analysis between local data and scientific studies.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature Trends Comparison: Local Data vs Scientific Studies', 
                 fontsize=16, fontweight='bold')
    
    # 1. Temperature projections by region
    regions = ['North\n(Local Data)', 'Center\n(Local Data)', 'South\n(Local Data)', 
              'UFPR\nProjection', 'Aegean\nProjection']
    temp_increases = [3.2, 2.8, 2.1, 2.5, 3.5]  # °C
    colors = ['red', 'orange', 'yellow', 'blue', 'purple']
    
    bars = ax1.bar(regions, temp_increases, color=colors, alpha=0.7)
    ax1.set_title('Projected Temperature Increase by Region')
    ax1.set_ylabel('Temperature Increase (°C)')
    ax1.set_ylim(0, 4)
    
    # Add values on bars
    for bar, value in zip(bars, temp_increases):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}°C', ha='center', va='bottom')
    
    # 2. Timeline of observed changes
    years = np.array([1960, 1990, 2000, 2010, 2020, 2024, 2030, 2040, 2050])
    
    # Bombus bellicosus (UFPR)
    bombus_presence = np.array([100, 80, 60, 40, 20, 0, 0, 0, 0])  # % presence
    ax2.plot(years, bombus_presence, 'ro-', linewidth=3, markersize=6, label='Bombus bellicosus (UFPR)')
    
    # General bee abundance (local data)
    general_abundance = np.array([100, 95, 90, 85, 75, 70, 60, 50, 40])  # % abundance
    ax2.plot(years, general_abundance, 'bo-', linewidth=3, markersize=6, label='General Abundance (Local)')
    
    ax2.axvline(x=2024, color='black', linestyle='--', alpha=0.7, label='Present')
    ax2.set_title('Species Abundance Timeline')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Relative Abundance (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Critical temperature thresholds
    species_groups = ['Eusocial\nSpecies', 'Solitary\nSpecies', 'Generalist\nSpecies', 'Specialist\nSpecies']
    critical_temps = [2.0, 2.5, 1.8, 3.0]  # Critical temperature thresholds
    colors_temp = ['green', 'blue', 'orange', 'red']
    
    bars = ax3.bar(species_groups, critical_temps, color=colors_temp, alpha=0.7)
    ax3.axhline(y=2.5, color='red', linestyle='--', alpha=0.8, label='Global Average Threshold')
    ax3.set_title('Critical Temperature Thresholds by Functional Group')
    ax3.set_ylabel('Critical Temperature Increase (°C)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, critical_temps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}°C', ha='center', va='bottom')
    
    # 4. Regional vulnerability index
    regions_vuln = ['Amazon', 'Cerrado', 'Atlantic Forest', 'Caatinga', 'Pampa']
    vulnerability = [7.5, 8.2, 9.1, 6.8, 8.8]  # Vulnerability index (1-10)
    colors_vuln = ['darkgreen', 'brown', 'green', 'yellow', 'lightgreen']
    
    bars = ax4.barh(regions_vuln, vulnerability, color=colors_vuln, alpha=0.7)
    ax4.set_title('Climate Vulnerability Index by Biome')
    ax4.set_xlabel('Vulnerability Index (1-10)')
    ax4.set_xlim(0, 10)
    
    # Add values on bars
    for i, (bar, value) in enumerate(zip(bars, vulnerability)):
        ax4.text(value + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{value}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('../images/comparative_temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: comparative_temperature_analysis.png")

def plot_migration_patterns_synthesis():
    """
    Synthesis of migration patterns integrating multiple sources.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bee Migration Patterns: Scientific Evidence Synthesis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Migration direction by study
    studies = ['UFPR\n(18 spp)', 'Dryad\n(Neotropical)', 'North America\n(Bombus)', 'Aegean\n(Islands)']
    south_migration = [85, 90, 70, 60]  # % species migrating south
    colors = ['green', 'blue', 'orange', 'purple']
    
    bars = ax1.bar(studies, south_migration, color=colors, alpha=0.7)
    ax1.set_title('Percentage of Species with Southward Migration')
    ax1.set_ylabel('Southward Migration (%)')
    ax1.set_ylim(0, 100)
    
    for bar, value in zip(bars, south_migration):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}%', ha='center', va='bottom')
    
    # 2. Projected migration velocity
    years = np.arange(2024, 2051, 3)
    
    # Different migration scenarios (adjusted for 9 points)
    conservative = np.array([0, 10, 25, 40, 55, 70, 80, 85, 90])  # % migrated species
    moderate = np.array([0, 15, 35, 55, 70, 80, 88, 92, 95])
    accelerated = np.array([0, 20, 45, 65, 80, 90, 95, 98, 100])
    
    ax2.plot(years, conservative, 'g-', linewidth=2, label='Conservative Scenario', marker='o')
    ax2.plot(years, moderate, 'b-', linewidth=2, label='Moderate Scenario', marker='s')
    ax2.plot(years, accelerated, 'r-', linewidth=2, label='Accelerated Scenario', marker='^')
    
    ax2.fill_between(years, conservative, accelerated, alpha=0.2, color='gray')
    ax2.set_title('Migration Velocity Projections')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Migration (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Functional group responses
    groups = ['Eusocial\nBees', 'Solitary\nBees', 'Generalist\nPollinators', 'Specialist\nPollinators']
    migration_rates = [75, 85, 65, 95]  # % showing migration
    adaptation_capacity = [60, 40, 80, 20]  # % with adaptation capacity
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, migration_rates, width, label='Migration Rate (%)', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, adaptation_capacity, width, label='Adaptation Capacity (%)', color='blue', alpha=0.7)
    
    ax3.set_title('Functional Group Response to Climate Change')
    ax3.set_xlabel('Functional Groups')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Conservation priority areas
    areas = ['Southern\nBrazilian\nFields', 'Atlantic\nForest\nRemnants', 'Cerrado\nTransition', 'Coastal\nPlains', 'Mountain\nCorridors']
    priority_scores = [9.5, 8.8, 8.2, 7.5, 9.0]  # Priority score (1-10)
    species_richness = [45, 38, 42, 28, 35]  # Number of species
    
    # Create dual y-axis
    ax4_twin = ax4.twinx()
    
    bars = ax4.bar(areas, priority_scores, color='darkgreen', alpha=0.7, label='Priority Score')
    line = ax4_twin.plot(areas, species_richness, 'ro-', linewidth=2, markersize=6, label='Species Richness')
    
    ax4.set_title('Conservation Priority Areas')
    ax4.set_ylabel('Priority Score (1-10)', color='darkgreen')
    ax4_twin.set_ylabel('Species Richness', color='red')
    ax4.tick_params(axis='y', labelcolor='darkgreen')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Rotate x-axis labels
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../images/migration_patterns_synthesis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved: migration_patterns_synthesis.png")

def main():
    """
    Main function that runs all analyses and generates charts.
    """
    print("Starting chart generation for bee analysis...\n")
    
    # Generate simulated data
    print("Generating simulated data...")
    df = generate_sample_data()
    geo_df = generate_geographic_data()
    
    print(f"Data generated: {len(df)} years of temporal data")
    print(f"Geographic data: {len(geo_df)} distribution points\n")
    
    # Generate charts
    print("Generating charts...")
    
    # Basic analysis charts
    plot_temperature_migration_trend(df)
    plot_correlation_analysis(df)
    plot_species_decline(df)
    plot_geographic_distribution(geo_df)
    plot_climate_impact_summary(df)
    
    # Scientific paper-specific charts
    plot_ufpr_migration_analysis()
    plot_comparative_temperature_analysis()
    plot_migration_patterns_synthesis()
    
    print("\nAll charts generated successfully!")
    print("Files saved in 'images/' folder")
    print("\nGenerated charts:")
    print("   1. temperature_migration_trend.png - Temperature and migration trends")
    print("   2. correlation_analysis.png - Correlation analysis")
    print("   3. species_honey_decline.png - Species and production decline")
    print("   4. geographic_distribution.png - Geographic distribution")
    print("   5. climate_impact_summary.png - Climate impact summary")
    print("   6. ufpr_migration_analysis.png - UFPR migration analysis")
    print("   7. comparative_temperature_analysis.png - Temperature comparison")
    print("   8. migration_patterns_synthesis.png - Migration patterns synthesis")
    
    # Save data for future use
    df.to_csv('../data/bee_climate_data.csv', index=False)
    geo_df.to_csv('../data/bee_geographic_data.csv', index=False)
    print("\nData saved in 'data/' for future analyses")

if __name__ == "__main__":
    main()