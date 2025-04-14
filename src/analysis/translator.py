def _translate_metric(metric: str) -> str:
    """Traduce nombres técnicos a español para reportes"""
    translations = {
        # Términos generales
        'team': 'Equipo',
        'metric': 'Métrica',
        'mean': 'Media',
        'median': 'Mediana',
        'mode': 'Moda',
        'std': 'DE',
        'IQR': 'IQR',
        'CV%': 'CV%',
        'min': 'Mínimo',
        'max': 'Máximo',
        
        # Métricas
        'goals': 'Goles',
        'shots': 'Tiros',
        'fouls': 'Faltas',
        'possession': 'Posesión (%)',
        'possession_time': 'Tiempo Posesión',
        
        # Percentiles
        'p25': 'P25',
        'p50': 'P50',
        'p75': 'P75',
        'p95': 'P95'
    }
    return translations.get(metric, metric.title())
    