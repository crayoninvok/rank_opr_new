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
        """
        Initialize the analyzer.
        
        Args:
            mohh (float): Maximum Operating Hours per Hour (default: 168.0)
        """
        self.mohh = mohh
        self.required_columns = ['vehicle_name', 'operator_name', 'status', 'duration']
        
    def parse_duration(self, dur_str: Any) -> float:
        """
        Parse duration string to hours as float with enhanced error handling.
        
        Args:
            dur_str: Duration string or any input type
            
        Returns:
            float: Duration in hours
        """
        if pd.isna(dur_str) or dur_str == '':
            return 0.0
            
        # Convert to string if not already
        dur_str = str(dur_str).strip()
        
        try:
            # Handle different duration formats
            if ':' in dur_str:
                # Format: HH:MM:SS.ssssss or HH:MM:SS
                parts = dur_str.split(':')
                if len(parts) < 2:
                    return 0.0
                    
                hours = int(parts[0]) if parts[0] else 0
                minutes = int(parts[1]) if parts[1] else 0
                
                # Handle seconds with optional microseconds
                seconds = 0
                microseconds = 0
                if len(parts) >= 3 and parts[2]:
                    if '.' in parts[2]:
                        sec_parts = parts[2].split('.')
                        seconds = int(sec_parts[0]) if sec_parts[0] else 0
                        # Handle microseconds (up to 6 digits)
                        if len(sec_parts) > 1:
                            ms_str = sec_parts[1].ljust(6, '0')[:6]
                            microseconds = int(ms_str)
                    else:
                        seconds = int(parts[2])
                
                # Convert to total hours
                td = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
                return td.total_seconds() / 3600.0
                
            else:
                # Try to parse as direct hours value
                return float(dur_str)
                
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Could not parse duration '{dur_str}': {e}")
            return 0.0
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if df.empty:
            return False, "DataFrame is empty"
            
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
            
        if len(df) == 0:
            return False, "No data rows found"
            
        return True, ""
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data with coal hauling specific status mapping.
        
        Args:
            df (pd.DataFrame): Raw input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df = df.copy()
        
        # Clean and standardize data
        df['vehicle_name'] = df['vehicle_name'].fillna('Unknown Vehicle').astype(str).str.strip()
        df['operator_name'] = df['operator_name'].fillna('Unknown Operator').astype(str).str.strip()
        df['status'] = df['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        # Parse duration
        df['duration_hours'] = df['duration'].apply(self.parse_duration)
        
        # Remove rows with invalid duration
        df = df[df['duration_hours'] >= 0]
        
        # Coal Hauling specific status mapping
        status_mapping = {
            # READY status -> OPERATION (vehicle is ready and operational)
            'READY': 'OPERATION',
            'COAL HAULING': 'OPERATION',
            
            # IDLE status -> DELAY (vehicle is idle but available)
            'IDLE': 'DELAY',
            'HUJAN': 'DELAY',  # Rain
            'DEMO': 'DELAY',   # Demo
            
            # DELAY status -> DELAY (all delay subcategories)
            'DELAY': 'DELAY',
            'ANTRIAN JEMBATAN TIMBANG': 'DELAY',  # Weighbridge queue
            'FATIGUE TEST (PENGECEKAN SAFETY)': 'DELAY',  # Fatigue test (safety check)
            'INTERNAL PROBLEM': 'DELAY',
            'JEMBATAN TIMBANG BERMASALAH': 'DELAY',  # Weighbridge problem
            'JETTY CROWDED (OVERCAPACITY)': 'DELAY',  # Jetty overcapacity
            'MAKAN/ISTIRAHAT/FATIGUE/IBADAH': 'DELAY',  # Meal/rest/fatigue/prayer
            'P2H': 'DELAY',  # Pre-shift inspection
            'P5M': 'DELAY',  # 5-minute safety talk
            'PEMBERSIHAN UNIT': 'DELAY',  # Unit cleaning
            'PENGISIAN BAHAN BAKAR': 'DELAY',  # Fuel filling
            'PERBAIKAN JALAN': 'DELAY',  # Road repair
            'SAFETY TALK': 'DELAY',
            'SIDE DUMP BERMASALAH': 'DELAY',  # Side dump problem
            'TUNGGU ALAT LOADING': 'DELAY',  # Waiting for loading equipment
            'TUNGGU LOGIN': 'DELAY',  # Waiting for login
            'TUNGGU OPERATOR DOUBLE TRAILER': 'DELAY',  # Waiting for double trailer operator
            'TUNGGU PETUGAS SIDE DUMP': 'DELAY',  # Waiting for side dump officer
            'TUNGGU STOCK BATUBARA COALPAD': 'DELAY',  # Waiting for coal stock at coalpad
            'TUNGGU TONGKANG JETTY': 'DELAY',  # Waiting for barge at jetty
            'WORKSHOP BERMASALAH (SELIP, STUCK)': 'DELAY',  # Workshop problem (slip, stuck)
            
            # BREAKDOWN status -> BREAKDOWN (all maintenance categories)
            'BREAKDOWN': 'BREAKDOWN',
            'PERIODIC INSPECTION': 'BREAKDOWN',
            'SCHEDULE MAINTENANCE': 'BREAKDOWN',
            'SCHEDULED MAINTENANCE': 'BREAKDOWN',
            'TIRE MAINTENANCE': 'BREAKDOWN',
            'UNSCHEDULE MAINTENANCE': 'BREAKDOWN',
            'UNSCHEDULED MAINTENANCE': 'BREAKDOWN',
            
            # Additional common variations
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
        
        # Log unique status values for debugging
        unique_statuses = df['status'].unique()
        logger.info(f"Unique status values after mapping: {list(unique_statuses)}")
        
        # Count records by status
        status_counts = df['status'].value_counts()
        logger.info(f"Status distribution: {dict(status_counts)}")
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame, group_by: str = 'vehicle_name') -> pd.DataFrame:
        """
        Calculate performance metrics for vehicles or operators.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame
            group_by (str): Column to group by ('vehicle_name' or 'operator_name')
            
        Returns:
            pd.DataFrame: DataFrame with calculated metrics
        """
        if group_by not in ['vehicle_name', 'operator_name']:
            raise ValueError("group_by must be 'vehicle_name' or 'operator_name'")
        
        # Group data
        grouped = df.groupby(group_by)
        
        # Calculate total hours by status for each group
        status_hours = df.groupby([group_by, 'status'])['duration_hours'].sum().unstack(fill_value=0)
        
        # Log available status columns for debugging
        logger.info(f"Available status columns: {list(status_hours.columns)}")
        
        # Ensure required status columns exist
        for status in ['BREAKDOWN', 'DELAY', 'OPERATION']:
            if status not in status_hours.columns:
                status_hours[status] = 0.0
                logger.info(f"Added missing status column: {status}")
        
        # Calculate metrics
        results = pd.DataFrame({
            group_by: status_hours.index,
            'breakdown_hours': status_hours['BREAKDOWN'],
            'delay_hours': status_hours['DELAY'],
            'mohh': status_hours['OPERATION'],
            'total_hours': status_hours.sum(axis=1)
        })
        
        # Log summary for debugging
        logger.info(f"Total breakdown hours: {results['breakdown_hours'].sum():.2f}")
        logger.info(f"Total delay hours: {results['delay_hours'].sum():.2f}")
        logger.info(f"Total operation hours: {results['mohh'].sum():.2f}")
        logger.info(f"Total hours across all statuses: {results['total_hours'].sum():.2f}")
        
        # Calculate Physical Availability (PA%)
        results['PA(%)'] = ((results['mohh'] - results['breakdown_hours']) / results['mohh']) * 100
        
        # Calculate Utilization Availability (UA%)
        available_hours = results['mohh'] - results['breakdown_hours']
        results['UA(%)'] = np.where(
            available_hours > 0,
            ((available_hours - results['delay_hours']) / available_hours) * 100,
            0.0
        )
        
        # Calculate additional metrics
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
        
        # Sort by availability score (descending) then by UA(%)
        results = results.sort_values(['availability_score', 'UA(%)'], ascending=[False, False]).reset_index(drop=True)
        results['rank'] = results.index + 1
        
        # Reorder columns
        column_order = ['rank', group_by, 'breakdown_hours', 'delay_hours', 'mohh', 
                       'total_hours', 'downtime_hours', 'PA(%)', 'UA(%)', 'efficiency(%)', 'availability_score']
        results = results[column_order]
        
        return results
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Args:
            results_df (pd.DataFrame): Results DataFrame
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
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
        """
        Get comprehensive analytics per status and per option.
        
        Args:
            df (pd.DataFrame): Raw DataFrame (before preprocessing)
            group_by (str): Column to group by
            
        Returns:
            Dict[str, Any]: Comprehensive status analytics
        """
        # Process the data to get duration in hours
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        # Define status categories with their options
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
        
        # 1. Overall status distribution (main categories)
        status_summary = df_analysis.groupby('status').agg({
            'duration_hours': ['sum', 'mean', 'count'],
            group_by: 'nunique'
        }).round(2)
        status_summary.columns = ['total_hours', 'avg_duration', 'frequency', 'unique_units']
        status_summary['percentage_of_total_time'] = (
            status_summary['total_hours'] / status_summary['total_hours'].sum() * 100
        ).round(2)
        
        analytics['status_summary'] = status_summary.to_dict('index')
        
        # 2. Detailed breakdown by unit and status
        unit_status_breakdown = df_analysis.groupby([group_by, 'status'])['duration_hours'].sum().unstack(fill_value=0)
        analytics['unit_status_breakdown'] = unit_status_breakdown.to_dict('index')
        
        # 3. Analytics per status category and their options
        category_analytics = {}
        
        for main_status, options in status_categories.items():
            category_data = {}
            
            # Main status analytics
            main_status_data = df_analysis[df_analysis['status'] == main_status]
            if not main_status_data.empty:
                category_data['main_status'] = {
                    'total_hours': main_status_data['duration_hours'].sum().round(2),
                    'frequency': len(main_status_data),
                    'avg_duration': main_status_data['duration_hours'].mean().round(2),
                    'unique_units': main_status_data[group_by].nunique(),
                    'percentage_of_total': (main_status_data['duration_hours'].sum() / 
                                          df_analysis['duration_hours'].sum() * 100).round(2)
                }
                
                # Unit-wise breakdown for this status
                unit_breakdown = main_status_data.groupby(group_by)['duration_hours'].sum().to_dict()
                category_data['unit_breakdown'] = {k: round(v, 2) for k, v in unit_breakdown.items()}
            
            # Options analytics for this status
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
                        'percentage_of_status_category': (
                            option_data['duration_hours'].sum() / max(status_total_hours, 0.001) * 100
                        ).round(2) if status_total_hours > 0 else 0,
                        'percentage_of_total': (
                            option_data['duration_hours'].sum() / df_analysis['duration_hours'].sum() * 100
                        ).round(2)
                    }
                    
                    # Unit-wise breakdown for this option
                    unit_breakdown = option_data.groupby(group_by)['duration_hours'].sum().to_dict()
                    options_data[option]['unit_breakdown'] = {k: round(v, 2) for k, v in unit_breakdown.items()}
            
            if options_data:
                category_data['options'] = options_data
                
            category_analytics[main_status] = category_data
        
        analytics['category_analytics'] = category_analytics
        
        # 4. Top issues analysis (most time-consuming statuses/options)
        all_statuses = df_analysis.groupby('status')['duration_hours'].sum().sort_values(ascending=False)
        analytics['top_time_consumers'] = {
            status: {
                'hours': round(hours, 2),
                'percentage': round(hours / df_analysis['duration_hours'].sum() * 100, 2)
            }
            for status, hours in all_statuses.head(10).items()
        }
        
        # 5. Unit performance comparison - FIXED: Handle empty status groups
        unit_performance = {}
        for unit in df_analysis[group_by].unique():
            unit_data = df_analysis[df_analysis[group_by] == unit]
            status_hours = unit_data.groupby('status')['duration_hours'].sum()
            
            # FIX: Check if status_hours is empty before calling idxmax()
            if len(status_hours) > 0:
                most_common = status_hours.idxmax()
            else:
                most_common = 'N/A'
            
            unit_performance[unit] = {
                'total_hours': unit_data['duration_hours'].sum().round(2),
                'status_distribution': status_hours.round(2).to_dict(),
                'most_common_status': most_common,
                'total_records': len(unit_data)
            }
        
        analytics['unit_performance'] = unit_performance
        
        # 6. Summary statistics
        analytics['summary_stats'] = {
            'total_records': len(df_analysis),
            'total_hours': df_analysis['duration_hours'].sum().round(2),
            'unique_units': df_analysis[group_by].nunique(),
            'unique_statuses': df_analysis['status'].nunique(),
            'avg_duration_per_record': df_analysis['duration_hours'].mean().round(2),
            'date_range': {
                'start': df_analysis.index.min() if hasattr(df_analysis.index, 'min') else 'N/A',
                'end': df_analysis.index.max() if hasattr(df_analysis.index, 'max') else 'N/A'
            } if 'date' in df_analysis.columns else 'No date column found'
        }
        
        return analytics
    
    def print_status_analytics(self, analytics: Dict[str, Any]) -> None:
        """
        Print formatted status analytics report.
        
        Args:
            analytics (Dict[str, Any]): Analytics from get_status_analytics()
        """
        print("=" * 80)
        print("COMPREHENSIVE STATUS ANALYTICS REPORT")
        print("=" * 80)
        
        # Summary
        print(f"\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        summary = analytics['summary_stats']
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Total Hours: {summary['total_hours']:,.2f}")
        print(f"Unique Units: {summary['unique_units']}")
        print(f"Unique Statuses: {summary['unique_statuses']}")
        print(f"Average Duration per Record: {summary['avg_duration_per_record']:.2f} hours")
        
        # Status Summary
        print(f"\n📋 STATUS OVERVIEW")
        print("-" * 40)
        for status, data in analytics['status_summary'].items():
            print(f"{status}:")
            print(f"  Total Hours: {data['total_hours']:,.2f} ({data['percentage_of_total_time']:.1f}%)")
            print(f"  Frequency: {data['frequency']:,} records")
            print(f"  Average Duration: {data['avg_duration']:.2f} hours")
            print(f"  Units Affected: {data['unique_units']}")
            print()
        
        # Detailed Category Analytics
        print(f"\n🔍 DETAILED CATEGORY ANALYTICS")
        print("-" * 40)
        
        for category, data in analytics['category_analytics'].items():
            print(f"\n{category.upper()} CATEGORY:")
            
            if 'main_status' in data:
                main = data['main_status']
                print(f"  Main Status Total: {main['total_hours']:,.2f} hours ({main['percentage_of_total']:.1f}%)")
            
            if 'options' in data:
                print(f"  Options Breakdown:")
                for option, option_data in data['options'].items():
                    print(f"    • {option}:")
                    print(f"      Hours: {option_data['total_hours']:,.2f} ({option_data['percentage_of_total']:.1f}% total, {option_data['percentage_of_status_category']:.1f}% of category)")
                    print(f"      Frequency: {option_data['frequency']:,} records")
                    print(f"      Avg Duration: {option_data['avg_duration']:.2f} hours")
                    print(f"      Units: {option_data['unique_units']}")
        
        # Top Time Consumers
        print(f"\n⏰ TOP TIME CONSUMERS")
        print("-" * 40)
        for i, (status, data) in enumerate(analytics['top_time_consumers'].items(), 1):
            print(f"{i:2d}. {status}: {data['hours']:,.2f} hours ({data['percentage']:.1f}%)")
        
        print("=" * 80)
    
    def get_option_distribution(self, df: pd.DataFrame, group_by: str = 'vehicle_name') -> Dict[str, Any]:
        """
        Get detailed distribution analysis per option with comprehensive breakdowns.
        
        Args:
            df (pd.DataFrame): Raw DataFrame (before preprocessing)
            group_by (str): Column to group by
            
        Returns:
            Dict[str, Any]: Option-level distribution analytics
        """
        # Process the data to get duration in hours
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        # Define all options by category
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
        
        # Analyze each option
        for category, options in status_options.items():
            # Include main status if it exists in data
            all_statuses_in_category = [category] + options
            
            for status in all_statuses_in_category:
                status_data = df_analysis[df_analysis['status'] == status]
                
                if not status_data.empty:
                    # Overall statistics
                    status_total_hours = status_data['duration_hours'].sum()
                    
                    option_distribution[status] = {
                        'category': category,
                        'total_hours': round(status_total_hours, 2),
                        'percentage_of_total': round((status_total_hours / total_hours) * 100, 2),
                        'frequency': len(status_data),
                        'avg_duration_hours': round(status_data['duration_hours'].mean(), 2),
                        'unique_units_affected': status_data[group_by].nunique(),
                        
                        # Unit-wise breakdown
                        'unit_breakdown': {},
                        'unit_percentages': {},
                        
                        # Time distribution
                        'min_duration': round(status_data['duration_hours'].min(), 2),
                        'max_duration': round(status_data['duration_hours'].max(), 2),
                        'median_duration': round(status_data['duration_hours'].median(), 2),
                        
                        # Affected units details
                        'units_affected': list(status_data[group_by].unique())
                    }
                    
                    # Calculate per-unit breakdown
                    unit_breakdown = status_data.groupby(group_by)['duration_hours'].agg(['sum', 'count', 'mean']).round(2)
                    
                    for unit in unit_breakdown.index:
                        unit_total = unit_breakdown.loc[unit, 'sum']
                        unit_count = unit_breakdown.loc[unit, 'count']
                        unit_avg = unit_breakdown.loc[unit, 'mean']
                        
                        option_distribution[status]['unit_breakdown'][unit] = {
                            'hours': unit_total,
                            'frequency': int(unit_count),
                            'avg_duration': unit_avg,
                            'percentage_of_option_total': round((unit_total / status_total_hours) * 100, 2)
                        }
        
        # Sort by total hours (descending)
        option_distribution = dict(sorted(
            option_distribution.items(), 
            key=lambda x: x[1]['total_hours'], 
            reverse=True
        ))
        
        return option_distribution
    
    def print_option_distribution(self, option_distribution: Dict[str, Any], top_n: int = None) -> None:
        """
        Print formatted option distribution report.
        
        Args:
            option_distribution (Dict[str, Any]): Option distribution from get_option_distribution()
            top_n (int): Show only top N options (None for all)
        """
        print("=" * 100)
        print("DETAILED OPTION DISTRIBUTION ANALYSIS")
        print("=" * 100)
        
        options_to_show = list(option_distribution.items())
        if top_n:
            options_to_show = options_to_show[:top_n]
            print(f"\n🔝 TOP {top_n} OPTIONS BY TOTAL HOURS")
        else:
            print(f"\n📊 ALL OPTIONS DISTRIBUTION ({len(options_to_show)} total)")
        
        print("-" * 100)
        
        for rank, (option, data) in enumerate(options_to_show, 1):
            print(f"\n{rank:2d}. {option} [{data['category']} CATEGORY]")
            print(f"    {'─' * 60}")
            print(f"    📊 Total Hours: {data['total_hours']:,.2f} ({data['percentage_of_total']:.1f}% of all time)")
            print(f"    🔄 Frequency: {data['frequency']:,} occurrences")
            print(f"    ⏱️  Average Duration: {data['avg_duration_hours']:.2f} hours")
            print(f"    🚛 Units Affected: {data['unique_units_affected']} units")
            print(f"    📈 Duration Range: {data['min_duration']:.2f}h - {data['max_duration']:.2f}h (median: {data['median_duration']:.2f}h)")
            
            # Show top 5 most affected units for this option
            if data['unit_breakdown']:
                print(f"    🎯 Most Affected Units:")
                unit_sorted = sorted(data['unit_breakdown'].items(), 
                                   key=lambda x: x[1]['hours'], reverse=True)
                for unit, unit_data in unit_sorted[:5]:  # Top 5 units
                    print(f"       • {unit}: {unit_data['hours']:.2f}h ({unit_data['percentage_of_option_total']:.1f}%) "
                          f"- {unit_data['frequency']} times, avg {unit_data['avg_duration']:.2f}h")
                
                if len(unit_sorted) > 5:
                    print(f"       ... and {len(unit_sorted) - 5} more units")
            
            print()
    
    def get_option_summary_table(self, option_distribution: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary table of option distribution for easy export/analysis.
        
        Args:
            option_distribution (Dict[str, Any]): Option distribution data
            
        Returns:
            pd.DataFrame: Summary table
        """
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
                'median_duration': data['median_duration'],
                'most_affected_unit': max(data['unit_breakdown'].items(), 
                                        key=lambda x: x[1]['hours'])[0] if data['unit_breakdown'] else 'N/A'
            })
        
        return pd.DataFrame(summary_data)
    
    def get_category_vs_options_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare main categories vs their options distribution.
        
        Args:
            df (pd.DataFrame): Raw DataFrame
            
        Returns:
            pd.DataFrame: Category vs options comparison
        """
        df_analysis = df.copy()
        df_analysis['duration_hours'] = df_analysis['duration'].apply(self.parse_duration)
        df_analysis['status'] = df_analysis['status'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        
        status_mapping = {
            'READY': 'OPERATION',
            'COAL HAULING': 'OPERATION',
            'IDLE': 'DELAY',
            'HUJAN': 'DELAY',
            'DEMO': 'DELAY',
            'DELAY': 'DELAY'
        }
        
        # Map delay options
        delay_options = [
            'ANTRIAN JEMBATAN TIMBANG', 'FATIGUE TEST (PENGECEKAN SAFETY)',
            'INTERNAL PROBLEM', 'JEMBATAN TIMBANG BERMASALAH',
            'JETTY CROWDED (OVERCAPACITY)', 'MAKAN/ISTIRAHAT/FATIGUE/IBADAH',
            'P2H', 'P5M', 'PEMBERSIHAN UNIT', 'PENGISIAN BAHAN BAKAR',
            'PERBAIKAN JALAN', 'SAFETY TALK', 'SIDE DUMP BERMASALAH',
            'TUNGGU ALAT LOADING', 'TUNGGU LOGIN', 'TUNGGU OPERATOR DOUBLE TRAILER',
            'TUNGGU PETUGAS SIDE DUMP', 'TUNGGU STOCK BATUBARA COALPAD',
            'TUNGGU TONGKANG JETTY', 'WORKSHOP BERMASALAH (SELIP, STUCK)'
        ]
        
        breakdown_options = [
            'PERIODIC INSPECTION', 'SCHEDULE MAINTENANCE',
            'TIRE MAINTENANCE', 'UNSCHEDULE MAINTENANCE'
        ]
        
        for option in delay_options:
            status_mapping[option] = 'DELAY'
        for option in breakdown_options:
            status_mapping[option] = 'BREAKDOWN'
        
        # Create comparison
        comparison_data = []
        total_hours = df_analysis['duration_hours'].sum()
        
        # Group by mapped categories
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
            
            # Add individual options for this category
            category_statuses = df_analysis[df_analysis['mapped_status'] == category]
            option_breakdown = category_statuses.groupby('status')['duration_hours'].agg(['sum', 'count'])
            
            for status, data in option_breakdown.iterrows():
                option_hours = data['sum']
                option_percentage = (option_hours / total_hours * 100) if total_hours > 0 else 0
                option_freq = data['count']
                
                comparison_data.append({
                    'type': 'OPTION',
                    'name': f"  └─ {status}",
                    'total_hours': round(option_hours, 2),
                    'percentage': round(option_percentage, 2),
                    'frequency': int(option_freq)
                })
        
        return pd.DataFrame(comparison_data)

def analyze_excel_file_comprehensive(excel_file: str, group_by: str = 'vehicle_name', mohh: float = 168.0) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Comprehensive analysis function that returns metrics, summary, and detailed status analytics.
    
    Args:
        excel_file (str): Path to Excel file
        group_by (str): Column to group by
        mohh (float): Maximum operating hours per hour
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]: (results_df, summary_stats, status_analytics)
    """
    try:
        # Initialize analyzer
        analyzer = VehicleMetricsAnalyzer(mohh=mohh)
        
        # Load data
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded {len(df)} rows from {excel_file}")
        
        # Validate data
        is_valid, error_msg = analyzer.validate_data(df)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get detailed status analytics BEFORE preprocessing (to preserve original statuses)
        status_analytics = analyzer.get_status_analytics(df, group_by)
        
        # Preprocess data
        df_processed = analyzer.preprocess_data(df)
        logger.info(f"Processed data: {len(df_processed)} valid rows")
        
        # Calculate metrics
        results = analyzer.calculate_metrics(df_processed, group_by)
        
        # Get summary statistics
        summary = analyzer.get_summary_statistics(results)
        
        return results, summary, status_analytics
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file '{excel_file}' not found.")
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise

def analyze_excel_file(excel_file: str, group_by: str = 'vehicle_name', mohh: float = 168.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to analyze Excel file and return results.
    
    Args:
        excel_file (str): Path to Excel file
        group_by (str): Column to group by
        mohh (float): Maximum operating hours per hour
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (results_df, summary_stats)
    """
    try:
        # Initialize analyzer
        analyzer = VehicleMetricsAnalyzer(mohh=mohh)
        
        # Load data
        df = pd.read_excel(excel_file)
        logger.info(f"Loaded {len(df)} rows from {excel_file}")
        
        # Validate data
        is_valid, error_msg = analyzer.validate_data(df)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Preprocess data
        df_processed = analyzer.preprocess_data(df)
        logger.info(f"Processed data: {len(df_processed)} valid rows")
        
        # Calculate metrics
        results = analyzer.calculate_metrics(df_processed, group_by)
        
        # Get summary statistics
        summary = analyzer.get_summary_statistics(results)
        
        return results, summary
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file '{excel_file}' not found.")
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise

# Backward compatibility function
def calculate_metrics(excel_file: str = 'status_duration.xlsx', group_by: str = 'vehicle_name') -> pd.DataFrame:
    """
    Backward compatibility function for the original calculate_metrics.
    
    Args:
        excel_file (str): Path to Excel file
        group_by (str): Column to group by
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    results, _ = analyze_excel_file(excel_file, group_by)
    return results