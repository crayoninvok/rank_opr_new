import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleMetricsAnalyzer:
    """Enhanced class for analyzing vehicle and operator performance metrics - Coal Hauling Operations."""
    
    def __init__(self, mohh: float = 168.0):
        """Initialize the analyzer."""
        self.mohh = mohh
        self.required_columns = ['vehicle_name', 'operator_name', 'status', 'duration']
        
    def parse_duration(self, dur_str: Any) -> float:
        """Parse duration string to hours as float with enhanced error handling."""
        if pd.isna(dur_str) or dur_str == '':
            return 0.0
            
        dur_str = str(dur_str).strip()
        
        try:
            if ':' in dur_str:
                parts = dur_str.split(':')
                if len(parts) < 2:
                    return 0.0
                    
                hours = int(parts[0]) if parts[0] else 0
                minutes = int(parts[1]) if parts[1] else 0
                
                seconds = 0
                microseconds = 0
                if len(parts) >= 3 and parts[2]:
                    if '.' in parts[2]:
                        sec_parts = parts[2].split('.')
                        seconds = int(sec_parts[0]) if sec_parts[0] else 0
                        if len(sec_parts) > 1:
                            ms_str = sec_parts[1].ljust(6, '0')[:6]
                            microseconds = int(ms_str)
                    else:
                        seconds = int(parts[2])
                
                td = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
                return td.total_seconds() / 3600.0
            else:
                return float(dur_str)
                
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Could not parse duration '{dur_str}': {e}")
            return 0.0
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the input DataFrame."""
        if df.empty:
            return False, "DataFrame is empty"
            
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
            
        if len(df) == 0:
            return False, "No data rows found"
            
        return True, ""
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data with coal hauling specific status mapping."""
        df = df.copy()
        
        # Clean and standardize data
        df['vehicle_name'] = df['vehicle_name'].fillna('Unknown Vehicle').astype(str).str.strip()
        df['operator_name'] = df['operator_name'].fillna('Unknown Operator').astype(str).str.strip()
        df['status'] = df['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        # Parse duration BEFORE any other operations
        df['duration_hours'] = df['duration'].apply(self.parse_duration)
        
        # Remove rows with invalid duration
        df = df[df['duration_hours'] >= 0].copy()
        
        # Coal Hauling specific status mapping
        status_mapping = {
            # READY status -> OPERATION
            'READY': 'OPERATION',
            'COAL HAULING': 'OPERATION',
            
            # IDLE status -> DELAY
            'IDLE': 'DELAY',
            'HUJAN': 'DELAY',
            'DEMO': 'DELAY',
            
            # DELAY status and all subcategories
            'DELAY': 'DELAY',
            'ANTRIAN JEMBATAN TIMBANG': 'DELAY',
            'FATIGUE TEST (PENGECEKAN SAFETY)': 'DELAY',
            'INTERNAL PROBLEM': 'DELAY',
            'JEMBATAN TIMBANG BERMASALAH': 'DELAY',
            'JETTY CROWDED (OVERCAPACITY)': 'DELAY',
            'MAKAN/ISTIRAHAT/FATIGUE/IBADAH': 'DELAY',
            'P2H': 'DELAY',
            'P5M': 'DELAY',
            'PEMBERSIHAN UNIT': 'DELAY',
            'PENGISIAN BAHAN BAKAR': 'DELAY',
            'PERBAIKAN JALAN': 'DELAY',
            'SAFETY TALK': 'DELAY',
            'SIDE DUMP BERMASALAH': 'DELAY',
            'TUNGGU ALAT LOADING': 'DELAY',
            'TUNGGU LOGIN': 'DELAY',
            'TUNGGU OPERATOR DOUBLE TRAILER': 'DELAY',
            'TUNGGU PETUGAS SIDE DUMP': 'DELAY',
            'TUNGGU STOCK BATUBARA COALPAD': 'DELAY',
            'TUNGGU TONGKANG JETTY': 'DELAY',
            'WORKSHOP BERMASALAH (SELIP, STUCK)': 'DELAY',
            
            # BREAKDOWN status and all maintenance categories
            'BREAKDOWN': 'BREAKDOWN',
            'PERIODIC INSPECTION': 'BREAKDOWN',
            'SCHEDULE MAINTENANCE': 'BREAKDOWN',
            'SCHEDULED MAINTENANCE': 'BREAKDOWN',
            'TIRE MAINTENANCE': 'BREAKDOWN',
            'UNSCHEDULE MAINTENANCE': 'BREAKDOWN',
            'UNSCHEDULED MAINTENANCE': 'BREAKDOWN',
            'BREAK DOWN': 'BREAKDOWN',
            'BREAK-DOWN': 'BREAKDOWN',
            'BREAK_DOWN': 'BREAKDOWN',
            'DOWN': 'BREAKDOWN',
            'BROKEN': 'BREAKDOWN',
            'BROKEN DOWN': 'BREAKDOWN',
            'OUT OF SERVICE': 'BREAKDOWN',
            'MAINTENANCE': 'BREAKDOWN',
            
            # Operation variations
            'OPERATIONAL': 'OPERATION',
            'OPERATING': 'OPERATION',
            'ACTIVE': 'OPERATION',
            'RUNNING': 'OPERATION',
            'IN SERVICE': 'OPERATION',
            'WORKING': 'OPERATION',
            'AVAILABLE': 'OPERATION',
            'ON DUTY': 'OPERATION',
            'SERVICE': 'OPERATION',
            'NORMAL': 'OPERATION',
            'OK': 'OPERATION',
            'GOOD': 'OPERATION',
            
            # Delay variations
            'DELAYED': 'DELAY',
            'WAITING': 'DELAY',
            'STANDBY': 'DELAY',
            'PAUSE': 'DELAY',
            'PAUSED': 'DELAY',
            'HOLD': 'DELAY',
            'PENDING': 'DELAY'
        }
        
        # Apply status mapping
        df['status'] = df['status'].replace(status_mapping)
        
        # Log for debugging
        logger.info(f"After preprocessing - Status distribution:")
        logger.info(f"{df['status'].value_counts()}")
        logger.info(f"Total hours by status:")
        logger.info(f"{df.groupby('status')['duration_hours'].sum()}")
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame, group_by: str = 'vehicle_name') -> pd.DataFrame:
        """Calculate performance metrics for vehicles or operators."""
        if group_by not in ['vehicle_name', 'operator_name']:
            raise ValueError("group_by must be 'vehicle_name' or 'operator_name'")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Calculating metrics grouped by: {group_by}")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique {group_by}: {df[group_by].nunique()}")
        
        # Verify required columns exist
        if 'duration_hours' not in df.columns:
            raise ValueError("'duration_hours' column missing - preprocess_data() must be called first")
        if 'status' not in df.columns:
            raise ValueError("'status' column missing")
        
        # Show status distribution before grouping
        logger.info(f"\nStatus distribution before grouping:")
        status_dist = df['status'].value_counts()
        for status, count in status_dist.items():
            total_hours = df[df['status'] == status]['duration_hours'].sum()
            logger.info(f"  {status}: {count} records, {total_hours:.2f} hours")
        
        # Group by entity and status, sum duration hours
        status_hours = df.groupby([group_by, 'status'])['duration_hours'].sum().unstack(fill_value=0)
        
        logger.info(f"\nAfter grouping and unstack:")
        logger.info(f"  Columns: {status_hours.columns.tolist()}")
        logger.info(f"  Shape: {status_hours.shape}")
        logger.info(f"  Column sums: {status_hours.sum().to_dict()}")
        
        # Ensure all required status columns exist
        for status in ['BREAKDOWN', 'DELAY', 'OPERATION']:
            if status not in status_hours.columns:
                status_hours[status] = 0.0
                logger.warning(f"  Added missing column: {status}")
        
        # Build results dataframe
        results = pd.DataFrame({
            group_by: status_hours.index,
            'breakdown_hours': status_hours['BREAKDOWN'],
            'delay_hours': status_hours['DELAY'],
            'mohh': status_hours['OPERATION'],
            'total_hours': status_hours.sum(axis=1)
        })
        
        logger.info(f"\nResults summary:")
        logger.info(f"  Breakdown hours: {results['breakdown_hours'].sum():.2f}")
        logger.info(f"  Delay hours: {results['delay_hours'].sum():.2f}")
        logger.info(f"  Operation hours: {results['mohh'].sum():.2f}")
        logger.info(f"  Total hours: {results['total_hours'].sum():.2f}")
        
        # Calculate PA%
        results['PA(%)'] = np.where(
            results['mohh'] > 0,
            ((results['mohh'] - results['breakdown_hours']) / results['mohh']) * 100,
            0.0
        )
        
        # Calculate UA%
        available_hours = results['mohh'] - results['breakdown_hours']
        results['UA(%)'] = np.where(
            available_hours > 0,
            ((available_hours - results['delay_hours']) / available_hours) * 100,
            0.0
        )
        
        # Additional metrics
        results['efficiency(%)'] = np.where(
            results['total_hours'] > 0,
            (results['mohh'] / results['total_hours']) * 100,
            0.0
        )
        
        results['downtime_hours'] = results['breakdown_hours'] + results['delay_hours']
        results['availability_score'] = (results['PA(%)'] + results['UA(%)']) / 2
        
        # Round numeric columns
        numeric_cols = ['breakdown_hours', 'delay_hours', 'mohh', 'total_hours', 
                       'downtime_hours', 'PA(%)', 'UA(%)', 'efficiency(%)', 'availability_score']
        for col in numeric_cols:
            results[col] = results[col].round(2)
        
        # Sort and rank
        results = results.sort_values(['availability_score', 'UA(%)'], ascending=[False, False]).reset_index(drop=True)
        results['rank'] = results.index + 1
        
        # Reorder columns
        column_order = ['rank', group_by, 'breakdown_hours', 'delay_hours', 'mohh', 
                       'total_hours', 'downtime_hours', 'PA(%)', 'UA(%)', 'efficiency(%)', 'availability_score']
        results = results[column_order]
        
        logger.info(f"{'='*60}\n")
        return results
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        return {
            'total_units': len(results_df),
            'avg_pa': results_df['PA(%)'].mean(),
            'avg_ua': results_df['UA(%)'].mean(),
            'avg_efficiency': results_df['efficiency(%)'].mean(),
            'total_breakdown_hours': results_df['breakdown_hours'].sum(),
            'total_delay_hours': results_df['delay_hours'].sum(),
            'total_mohh': results_df['mohh'].sum(),
            'best_performer': results_df.iloc[0][results_df.columns[1]] if len(results_df) > 0 else 'N/A',
            'worst_performer': results_df.iloc[-1][results_df.columns[1]] if len(results_df) > 0 else 'N/A'
        }
    
    def get_status_analytics(self, df: pd.DataFrame, group_by: str = 'vehicle_name') -> Dict[str, Any]:
        """Get comprehensive analytics per status and per option."""
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        status_categories = {
            'READY': ['COAL HAULING'],
            'IDLE': ['HUJAN', 'DEMO'],
            'DELAY': [
                'ANTRIAN JEMBATAN TIMBANG', 'FATIGUE TEST (PENGECEKAN SAFETY)',
                'INTERNAL PROBLEM', 'JEMBATAN TIMBANG BERMASALAH',
                'JETTY CROWDED (OVERCAPACITY)', 'MAKAN/ISTIRAHAT/FATIGUE/IBADAH',
                'P2H', 'P5M', 'PEMBERSIHAN UNIT', 'PENGISIAN BAHAN BAKAR',
                'PERBAIKAN JALAN', 'SAFETY TALK', 'SIDE DUMP BERMASALAH',
                'TUNGGU ALAT LOADING', 'TUNGGU LOGIN', 'TUNGGU OPERATOR DOUBLE TRAILER',
                'TUNGGU PETUGAS SIDE DUMP', 'TUNGGU STOCK BATUBARA COALPAD',
                'TUNGGU TONGKANG JETTY', 'WORKSHOP BERMASALAH (SELIP, STUCK)'
            ],
            'BREAKDOWN': [
                'PERIODIC INSPECTION', 'SCHEDULE MAINTENANCE',
                'TIRE MAINTENANCE', 'UNSCHEDULE MAINTENANCE'
            ]
        }
        
        analytics = {}
        
        # Overall status distribution
        status_summary = df_analysis.groupby('status').agg({
            'duration_hours': ['sum', 'mean', 'count'],
            group_by: 'nunique'
        }).round(2)
        status_summary.columns = ['total_hours', 'avg_duration', 'frequency', 'unique_units']
        status_summary['percentage_of_total_time'] = (
            status_summary['total_hours'] / status_summary['total_hours'].sum() * 100
        ).round(2)
        
        analytics['status_summary'] = status_summary.to_dict('index')
        analytics['unit_status_breakdown'] = df_analysis.groupby([group_by, 'status'])['duration_hours'].sum().unstack(fill_value=0).to_dict('index')
        
        # Category analytics
        category_analytics = {}
        for main_status, options in status_categories.items():
            category_data = {}
            main_status_data = df_analysis[df_analysis['status'] == main_status]
            
            if not main_status_data.empty:
                category_data['main_status'] = {
                    'total_hours': main_status_data['duration_hours'].sum().round(2),
                    'frequency': len(main_status_data),
                    'avg_duration': main_status_data['duration_hours'].mean().round(2),
                    'unique_units': main_status_data[group_by].nunique(),
                    'percentage_of_total': (main_status_data['duration_hours'].sum() / df_analysis['duration_hours'].sum() * 100).round(2)
                }
                category_data['unit_breakdown'] = {k: round(v, 2) for k, v in main_status_data.groupby(group_by)['duration_hours'].sum().to_dict().items()}
            
            options_data = {}
            status_total_hours = df_analysis[df_analysis['status'].isin([main_status] + options)]['duration_hours'].sum()
            
            for option in options:
                option_data = df_analysis[df_analysis['status'] == option]
                if not option_data.empty:
                    options_data[option] = {
                        'total_hours': option_data['duration_hours'].sum().round(2),
                        'frequency': len(option_data),
                        'avg_duration': option_data['duration_hours'].mean().round(2),
                        'unique_units': option_data[group_by].nunique(),
                        'percentage_of_status_category': (option_data['duration_hours'].sum() / max(status_total_hours, 0.001) * 100).round(2) if status_total_hours > 0 else 0,
                        'percentage_of_total': (option_data['duration_hours'].sum() / df_analysis['duration_hours'].sum() * 100).round(2),
                        'unit_breakdown': {k: round(v, 2) for k, v in option_data.groupby(group_by)['duration_hours'].sum().to_dict().items()}
                    }
            
            if options_data:
                category_data['options'] = options_data
            category_analytics[main_status] = category_data
        
        analytics['category_analytics'] = category_analytics
        
        # Top time consumers
        all_statuses = df_analysis.groupby('status')['duration_hours'].sum().sort_values(ascending=False)
        analytics['top_time_consumers'] = {
            status: {'hours': round(hours, 2), 'percentage': round(hours / df_analysis['duration_hours'].sum() * 100, 2)}
            for status, hours in all_statuses.head(10).items()
        }
        
        # Unit performance
        unit_performance = {}
        for unit in df_analysis[group_by].unique():
            unit_data = df_analysis[df_analysis[group_by] == unit]
            status_hours = unit_data.groupby('status')['duration_hours'].sum()
            
            unit_performance[unit] = {
                'total_hours': unit_data['duration_hours'].sum().round(2),
                'status_distribution': status_hours.round(2).to_dict(),
                'most_common_status': status_hours.idxmax() if len(status_hours) > 0 else 'N/A',
                'total_records': len(unit_data)
            }
        
        analytics['unit_performance'] = unit_performance
        analytics['summary_stats'] = {
            'total_records': len(df_analysis),
            'total_hours': df_analysis['duration_hours'].sum().round(2),
            'unique_units': df_analysis[group_by].nunique(),
            'unique_statuses': df_analysis['status'].nunique(),
            'avg_duration_per_record': df_analysis['duration_hours'].mean().round(2)
        }
        
        return analytics
    
    def get_option_distribution(self, df: pd.DataFrame, group_by: str = 'vehicle_name') -> Dict[str, Any]:
        """Get detailed distribution analysis per option."""
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        status_options = {
            'READY': ['COAL HAULING'],
            'IDLE': ['HUJAN', 'DEMO'],
            'DELAY': [
                'ANTRIAN JEMBATAN TIMBANG', 'FATIGUE TEST (PENGECEKAN SAFETY)',
                'INTERNAL PROBLEM', 'JEMBATAN TIMBANG BERMASALAH',
                'JETTY CROWDED (OVERCAPACITY)', 'MAKAN/ISTIRAHAT/FATIGUE/IBADAH',
                'P2H', 'P5M', 'PEMBERSIHAN UNIT', 'PENGISIAN BAHAN BAKAR',
                'PERBAIKAN JALAN', 'SAFETY TALK', 'SIDE DUMP BERMASALAH',
                'TUNGGU ALAT LOADING', 'TUNGGU LOGIN', 'TUNGGU OPERATOR DOUBLE TRAILER',
                'TUNGGU PETUGAS SIDE DUMP', 'TUNGGU STOCK BATUBARA COALPAD',
                'TUNGGU TONGKANG JETTY', 'WORKSHOP BERMASALAH (SELIP, STUCK)'
            ],
            'BREAKDOWN': [
                'PERIODIC INSPECTION', 'SCHEDULE MAINTENANCE',
                'TIRE MAINTENANCE', 'UNSCHEDULE MAINTENANCE'
            ]
        }
        
        total_hours = df_analysis['duration_hours'].sum()
        option_distribution = {}
        
        for category, options in status_options.items():
            all_statuses_in_category = [category] + options
            
            for status in all_statuses_in_category:
                status_data = df_analysis[df_analysis['status'] == status]
                
                if not status_data.empty:
                    status_total_hours = status_data['duration_hours'].sum()
                    
                    option_distribution[status] = {
                        'category': category,
                        'total_hours': round(status_total_hours, 2),
                        'percentage_of_total': round((status_total_hours / total_hours) * 100, 2),
                        'frequency': len(status_data),
                        'avg_duration_hours': round(status_data['duration_hours'].mean(), 2),
                        'unique_units_affected': status_data[group_by].nunique(),
                        'unit_breakdown': {},
                        'min_duration': round(status_data['duration_hours'].min(), 2),
                        'max_duration': round(status_data['duration_hours'].max(), 2),
                        'median_duration': round(status_data['duration_hours'].median(), 2),
                        'units_affected': list(status_data[group_by].unique())
                    }
                    
                    unit_breakdown = status_data.groupby(group_by)['duration_hours'].agg(['sum', 'count', 'mean']).round(2)
                    for unit in unit_breakdown.index:
                        option_distribution[status]['unit_breakdown'][unit] = {
                            'hours': unit_breakdown.loc[unit, 'sum'],
                            'frequency': int(unit_breakdown.loc[unit, 'count']),
                            'avg_duration': unit_breakdown.loc[unit, 'mean'],
                            'percentage_of_option_total': round((unit_breakdown.loc[unit, 'sum'] / status_total_hours) * 100, 2)
                        }
        
        return dict(sorted(option_distribution.items(), key=lambda x: x[1]['total_hours'], reverse=True))
    
    def get_option_summary_table(self, option_distribution: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary table of option distribution."""
        summary_data = []
        for option, data in option_distribution.items():
            summary_data.append({
                'rank': len(summary_data) + 1,
                'option': option,
                'category': data['category'],
                'total_hours': data['total_hours'],
                'percentage_of_total': data['percentage_of_total'],
                'frequency': data['frequency'],
                'avg_duration_hours': data['avg_duration_hours'],
                'unique_units_affected': data['unique_units_affected'],
                'min_duration': data['min_duration'],
                'max_duration': data['max_duration'],
                'median_duration': data['median_duration']
            })
        return pd.DataFrame(summary_data)
    
    def get_category_vs_options_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare main categories vs their options distribution."""
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        comparison_data = []
        total_hours = df_analysis['duration_hours'].sum()
        
        # Map statuses to categories
        status_mapping = {'READY': 'OPERATION', 'COAL HAULING': 'OPERATION', 'IDLE': 'DELAY', 'HUJAN': 'DELAY', 'DEMO': 'DELAY', 'DELAY': 'DELAY'}
        
        delay_options = ['ANTRIAN JEMBATAN TIMBANG', 'FATIGUE TEST (PENGECEKAN SAFETY)', 'INTERNAL PROBLEM', 'JEMBATAN TIMBANG BERMASALAH',
                        'JETTY CROWDED (OVERCAPACITY)', 'MAKAN/ISTIRAHAT/FATIGUE/IBADAH', 'P2H', 'P5M', 'PEMBERSIHAN UNIT', 'PENGISIAN BAHAN BAKAR',
                        'PERBAIKAN JALAN', 'SAFETY TALK', 'SIDE DUMP BERMASALAH', 'TUNGGU ALAT LOADING', 'TUNGGU LOGIN', 'TUNGGU OPERATOR DOUBLE TRAILER',
                        'TUNGGU PETUGAS SIDE DUMP', 'TUNGGU STOCK BATUBARA COALPAD', 'TUNGGU TONGKANG JETTY', 'WORKSHOP BERMASALAH (SELIP, STUCK)']
        
        breakdown_options = ['PERIODIC INSPECTION', 'SCHEDULE MAINTENANCE', 'TIRE MAINTENANCE', 'UNSCHEDULE MAINTENANCE']
        
        for option in delay_options:
            status_mapping[option] = 'DELAY'
        for option in breakdown_options:
            status_mapping[option] = 'BREAKDOWN'
        
        df_analysis['mapped_status'] = df_analysis['status'].map(status_mapping).fillna('OTHER')
        category_totals = df_analysis.groupby('mapped_status')['duration_hours'].sum()
        
        for category in ['OPERATION', 'DELAY', 'BREAKDOWN']:
            category_total = category_totals.get(category, 0)
            category_percentage = (category_total / total_hours * 100) if total_hours > 0 else 0
            
            comparison_data.append({
                'type': 'CATEGORY',
                'name': category,
                'total_hours': round(category_total, 2),
                'percentage': round(category_percentage, 2),
                'frequency': len(df_analysis[df_analysis['mapped_status'] == category])
            })
            
            category_statuses = df_analysis[df_analysis['mapped_status'] == category]
            option_breakdown = category_statuses.groupby('status')['duration_hours'].agg(['sum', 'count'])
            
            for status, data in option_breakdown.iterrows():
                comparison_data.append({
                    'type': 'OPTION',
                    'name': f"  └─ {status}",
                    'total_hours': round(data['sum'], 2),
                    'percentage': round((data['sum'] / total_hours * 100) if total_hours > 0 else 0, 2),
                    'frequency': int(data['count'])
                })
        
        return pd.DataFrame(comparison_data)

def analyze_excel_file_comprehensive(excel_file: str, group_by: str = 'vehicle_name', mohh: float = 168.0) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Comprehensive analysis function."""
    try:
        analyzer = VehicleMetricsAnalyzer(mohh=mohh)
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded {len(df)} rows from {excel_file}")
        
        is_valid, error_msg = analyzer.validate_data(df)
        if not is_valid:
            raise ValueError(error_msg)
        
        status_analytics = analyzer.get_status_analytics(df, group_by)
        df_processed = analyzer.preprocess_data(df)
        logger.info(f"Processed data: {len(df_processed)} valid rows")
        
        results = analyzer.calculate_metrics(df_processed, group_by)
        summary = analyzer.get_summary_statistics(results)
        
        return results, summary, status_analytics
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file '{excel_file}' not found.")
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise

def analyze_excel_file(excel_file: str, group_by: str = 'vehicle_name', mohh: float = 168.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Main function to analyze Excel file."""
    try:
        analyzer = VehicleMetricsAnalyzer(mohh=mohh)
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded {len(df)} rows from {excel_file}")
        
        is_valid, error_msg = analyzer.validate_data(df)
        if not is_valid:
            raise ValueError(error_msg)
        
        df_processed = analyzer.preprocess_data(df)
        logger.info(f"Processed data: {len(df_processed)} valid rows")
        
        results = analyzer.calculate_metrics(df_processed, group_by)
        summary = analyzer.get_summary_statistics(results)
        
        return results, summary
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file '{excel_file}' not found.")
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise

def calculate_metrics(excel_file: str = 'status_duration.xlsx', group_by: str = 'vehicle_name') -> pd.DataFrame:
    """Backward compatibility function."""
    results, _ = analyze_excel_file(excel_file, group_by)
    return results