import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleMetricsAnalyzer:
    """Enhanced class for analyzing vehicle and operator performance metrics."""
    
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
        Preprocess the input data.
        
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
        
        # Standardize status values - more comprehensive mapping
        status_mapping = {
            # Breakdown variations
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
            'READY': 'OPERATION',
            'ON DUTY': 'OPERATION',
            'SERVICE': 'OPERATION',
            'NORMAL': 'OPERATION',
            'OK': 'OPERATION',
            'GOOD': 'OPERATION',
            
            # Delay variations
            'DELAYED': 'DELAY',
            'WAITING': 'DELAY',
            'IDLE': 'DELAY',
            'STANDBY': 'DELAY',
            'PAUSE': 'DELAY',
            'PAUSED': 'DELAY',
            'HOLD': 'DELAY',
            'PENDING': 'DELAY'
        }
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
        
        # If no OPERATION data found, try to infer from other statuses or data patterns
        if status_hours['OPERATION'].sum() == 0:
            logger.warning("No OPERATION status found. Checking for alternative operational indicators...")
            
            # Check if there are other status values that might represent operations
            other_statuses = [col for col in status_hours.columns if col not in ['BREAKDOWN', 'DELAY', 'OPERATION']]
            if other_statuses:
                logger.info(f"Found other status values: {other_statuses}")
                
                # Sum all non-breakdown, non-delay statuses as operational
                operational_cols = [col for col in other_statuses if col not in ['UNKNOWN', 'NULL', 'NONE', '']]
                if operational_cols:
                    status_hours['OPERATION'] = status_hours[operational_cols].sum(axis=1)
                    logger.info(f"Converted {operational_cols} to OPERATION status")
        
        # Calculate metrics
        results = pd.DataFrame({
            group_by: status_hours.index,
            'breakdown_hours': status_hours['BREAKDOWN'],
            'delay_hours': status_hours['DELAY'],
            'operation_hours': status_hours['OPERATION'],
            'total_hours': status_hours.sum(axis=1)
        })
        
        # Log summary for debugging
        logger.info(f"Total breakdown hours: {results['breakdown_hours'].sum():.2f}")
        logger.info(f"Total delay hours: {results['delay_hours'].sum():.2f}")
        logger.info(f"Total operation hours: {results['operation_hours'].sum():.2f}")
        logger.info(f"Total hours across all statuses: {results['total_hours'].sum():.2f}")
        
        # Calculate Physical Availability (PA%)
        results['PA(%)'] = ((self.mohh - results['breakdown_hours']) / self.mohh) * 100
        
        # Calculate Utilization Availability (UA%)
        available_hours = self.mohh - results['breakdown_hours']
        results['UA(%)'] = np.where(
            available_hours > 0,
            ((available_hours - results['delay_hours']) / available_hours) * 100,
            0.0
        )
        
        # Calculate additional metrics
        results['efficiency(%)'] = np.where(
            results['total_hours'] > 0,
            (results['operation_hours'] / results['total_hours']) * 100,
            0.0
        )
        
        results['downtime_hours'] = results['breakdown_hours'] + results['delay_hours']
        results['availability_score'] = (results['PA(%)'] + results['UA(%)']) / 2
        
        # Round numeric columns
        numeric_cols = ['breakdown_hours', 'delay_hours', 'operation_hours', 'total_hours', 
                       'downtime_hours', 'PA(%)', 'UA(%)', 'efficiency(%)', 'availability_score']
        for col in numeric_cols:
            results[col] = results[col].round(2)
        
        # Sort by availability score (descending) then by UA(%)
        results = results.sort_values(['availability_score', 'UA(%)'], ascending=[False, False]).reset_index(drop=True)
        results['rank'] = results.index + 1
        
        # Reorder columns
        column_order = ['rank', group_by, 'breakdown_hours', 'delay_hours', 'operation_hours', 
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
            'total_operation_hours': results_df['operation_hours'].sum(),
            'best_performer': results_df.iloc[0][results_df.columns[1]] if len(results_df) > 0 else 'N/A',
            'worst_performer': results_df.iloc[-1][results_df.columns[1]] if len(results_df) > 0 else 'N/A'
        }

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