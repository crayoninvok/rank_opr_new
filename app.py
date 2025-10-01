import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import time
from analyze import analyze_excel_file_comprehensive, VehicleMetricsAnalyzer

# Page configuration
st.set_page_config(
    page_title="Vehicle & Operator Analytics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-danger { color: #dc3545; font-weight: bold; }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .option-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def safe_remove_temp_file(file_path, max_attempts=5, delay=0.1):
    """Safely remove temporary file with retry mechanism."""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.warning(f"Could not remove temporary file: {file_path}")
                return False
    return True

def create_summary_cards(summary_stats):
    """Create metric cards for summary statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Average Physical Availability",
            value=f"{summary_stats['avg_pa']:.1f}%",
            delta=f"{summary_stats['avg_pa'] - 85:.1f}% vs target" if summary_stats['avg_pa'] > 0 else None
        )
    
    with col2:
        st.metric(
            label="⚡ Average Utilization",
            value=f"{summary_stats['avg_ua']:.1f}%",
            delta=f"{summary_stats['avg_ua'] - 80:.1f}% vs target" if summary_stats['avg_ua'] > 0 else None
        )
    
    with col3:
        st.metric(
            label="📊 Average Efficiency",
            value=f"{summary_stats['avg_efficiency']:.1f}%",
            delta=f"{summary_stats['avg_efficiency'] - 75:.1f}% vs target" if summary_stats['avg_efficiency'] > 0 else None
        )
    
    with col4:
        st.metric(
            label="📈 Total Units Analyzed",
            value=f"{summary_stats['total_units']:,}"
        )

def create_performance_charts(results_df, ranking_type):
    """Create interactive charts for performance analysis."""
    
    # Prepare data
    entity_col = f"{ranking_type.lower()}_name"
    top_10 = results_df.head(10).copy()
    
    # 1. Availability Comparison Chart
    fig_availability = go.Figure()
    
    fig_availability.add_trace(go.Bar(
        name='Physical Availability (%)',
        x=top_10[entity_col],
        y=top_10['PA(%)'],
        marker_color='#2E86AB',
        yaxis='y1'
    ))
    
    fig_availability.add_trace(go.Bar(
        name='Utilization Availability (%)',
        x=top_10[entity_col],
        y=top_10['UA(%)'],
        marker_color='#A23B72',
        yaxis='y1'
    ))
    
    fig_availability.update_layout(
        title=f'Top 10 {ranking_type}s - Availability Metrics',
        xaxis_title=f'{ranking_type} Name',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    # 2. Hours Distribution Chart
    fig_hours = go.Figure()
    
    fig_hours.add_trace(go.Bar(
        name='Operation Hours',
        x=top_10[entity_col],
        y=top_10['operation_hours'],
        marker_color='#28a745'
    ))
    
    fig_hours.add_trace(go.Bar(
        name='Delay Hours',
        x=top_10[entity_col],
        y=top_10['delay_hours'],
        marker_color='#ffc107'
    ))
    
    fig_hours.add_trace(go.Bar(
        name='Breakdown Hours',
        x=top_10[entity_col],
        y=top_10['breakdown_hours'],
        marker_color='#dc3545'
    ))
    
    fig_hours.update_layout(
        title=f'Top 10 {ranking_type}s - Hours Distribution',
        xaxis_title=f'{ranking_type} Name',
        yaxis_title='Hours',
        barmode='stack',
        height=500,
        showlegend=True
    )
    
    # 3. Performance Scatter Plot
    fig_scatter = px.scatter(
        results_df, 
        x='PA(%)', 
        y='UA(%)',
        size='total_hours',
        color='availability_score',
        hover_data=[entity_col, 'efficiency(%)', 'rank'],
        title=f'{ranking_type} Performance Matrix',
        labels={'PA(%)': 'Physical Availability (%)', 'UA(%)': 'Utilization Availability (%)'},
        color_continuous_scale='RdYlGn'
    )
    
    fig_scatter.update_layout(height=500)
    
    return fig_availability, fig_hours, fig_scatter

def create_option_distribution_chart(option_distribution, top_n=15):
    """Create chart for option distribution analysis."""
    # Get top N options
    options_list = list(option_distribution.items())[:top_n]
    
    options = [opt for opt, _ in options_list]
    hours = [data['total_hours'] for _, data in options_list]
    percentages = [data['percentage_of_total'] for _, data in options_list]
    categories = [data['category'] for _, data in options_list]
    
    # Color mapping for categories
    color_map = {
        'OPERATION': '#28a745',
        'DELAY': '#ffc107', 
        'BREAKDOWN': '#dc3545',
        'READY': '#17a2b8',
        'IDLE': '#6c757d'
    }
    colors = [color_map.get(cat, '#6c757d') for cat in categories]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=options[::-1],  # Reverse for better display
        x=hours[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{p:.1f}%" for p in percentages[::-1]],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Hours: %{x:.1f}<br>Percentage: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Options by Total Hours',
        xaxis_title='Total Hours',
        yaxis_title='Options',
        height=600,
        showlegend=False,
        margin=dict(l=200)  # More space for option names
    )
    
    return fig

def create_category_breakdown_chart(option_distribution):
    """Create pie chart for category breakdown."""
    # Aggregate by category
    category_totals = {}
    for option, data in option_distribution.items():
        category = data['category']
        if category not in category_totals:
            category_totals[category] = 0
        category_totals[category] += data['total_hours']
    
    fig = px.pie(
        values=list(category_totals.values()),
        names=list(category_totals.keys()),
        title='Time Distribution by Category',
        color_discrete_map={
            'OPERATION': '#28a745',
            'DELAY': '#ffc107', 
            'BREAKDOWN': '#dc3545',
            'READY': '#17a2b8',
            'IDLE': '#6c757d'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def display_option_analytics(option_distribution, top_n=10):
    """Display option distribution analytics in a structured way."""
    st.markdown("### 🔍 Detailed Option Analysis")
    
    options_list = list(option_distribution.items())[:top_n]
    
    for rank, (option, data) in enumerate(options_list, 1):
        with st.container():
            # Main option info
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                category_emoji = {
                    'OPERATION': '✅', 'DELAY': '⏳', 'BREAKDOWN': '🔧',
                    'READY': '🚀', 'IDLE': '⏸️'
                }
                emoji = category_emoji.get(data['category'], '📋')
                st.write(f"**{rank}. {emoji} {option}**")
                st.caption(f"Category: {data['category']}")
            
            with col2:
                st.metric("Total Hours", f"{data['total_hours']:.1f}")
            
            with col3:
                st.metric("% of Total", f"{data['percentage_of_total']:.1f}%")
            
            with col4:
                st.metric("Frequency", f"{data['frequency']:,}")
            
            # Additional details in expandable section
            with st.expander(f"Details for {option}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Duration Statistics:**")
                    st.write(f"• Average: {data['avg_duration_hours']:.2f} hours")
                    st.write(f"• Minimum: {data['min_duration']:.2f} hours")
                    st.write(f"• Maximum: {data['max_duration']:.2f} hours")
                    st.write(f"• Median: {data['median_duration']:.2f} hours")
                
                with col2:
                    st.write("**Unit Impact:**")
                    st.write(f"• Units Affected: {data['unique_units_affected']}")
                    
                    if data['unit_breakdown']:
                        st.write("**Most Affected Units:**")
                        unit_sorted = sorted(data['unit_breakdown'].items(), 
                                           key=lambda x: x[1]['hours'], reverse=True)
                        for unit, unit_data in unit_sorted[:3]:  # Top 3
                            st.write(f"• {unit}: {unit_data['hours']:.1f}h ({unit_data['frequency']}x)")
            
            st.divider()

def style_dataframe(df):
    """Apply styling to the dataframe."""
    def highlight_performance(val):
        if isinstance(val, (int, float)):
            if val >= 90:
                return 'background-color: #d4edda; color: #155724'
            elif val >= 70:
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    # Apply styling to percentage columns
    percentage_cols = ['PA(%)', 'UA(%)', 'efficiency(%)', 'availability_score']
    styled_df = df.style.applymap(highlight_performance, subset=percentage_cols)
    
    # Format numeric columns
    format_dict = {
        'breakdown_hours': '{:.1f}',
        'delay_hours': '{:.1f}',
        'operation_hours': '{:.1f}',
        'total_hours': '{:.1f}',
        'downtime_hours': '{:.1f}',
        'PA(%)': '{:.1f}%',
        'UA(%)': '{:.1f}%',
        'efficiency(%)': '{:.1f}%',
        'availability_score': '{:.1f}'
    }
    
    styled_df = styled_df.format(format_dict)
    
    return styled_df

def main():
    """Main function to run the enhanced Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🚛 Vehicle & Operator Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Analysis type selection
        ranking_type = st.selectbox(
            "📊 Analysis Type",
            ["Vehicle", "Operator"],
            index=0,
            help="Choose whether to analyze vehicles or operators"
        )
        
        # MOHH configuration
        mohh = st.number_input(
            "⏰ Maximum Operating Hours per Week",
            min_value=1.0,
            max_value=300.0,
            value=168.0,
            step=1.0,
            help="Standard week has 168 hours (24h × 7 days)"
        )
        
        # Performance thresholds
        st.markdown("### 🎯 Performance Thresholds")
        pa_threshold = st.slider("Physical Availability Target (%)", 0, 100, 85)
        ua_threshold = st.slider("Utilization Availability Target (%)", 0, 100, 80)
        eff_threshold = st.slider("Efficiency Target (%)", 0, 100, 75)
        
        # Option analysis configuration
        st.markdown("### 🔍 Option Analysis")
        show_option_analysis = st.checkbox("Show Option Distribution Analysis", value=True)
        top_n_options = st.slider("Number of Top Options to Display", 5, 30, 15)
        
        # About section
        with st.expander("ℹ️ About This Tool"):
            st.write("""
            **Physical Availability (PA%)**: Percentage of time equipment is not broken down
            
            **Utilization Availability (UA%)**: Percentage of available time actually used
            
            **Efficiency (%)**: Percentage of total logged time spent in operation
            
            **Availability Score**: Average of PA% and UA%
            
            **Option Distribution**: Breakdown of specific issues within each status category
            """)
    
    # File upload section
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Upload your vehicle/operator status data file"
    )
    
    if uploaded_file is not None:
        temp_file_path = None
        try:
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            with st.expander("📋 File Information"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Save uploaded file temporarily with better context management
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            
            # Process data
            with st.spinner('🔄 Processing data...'):
                group_by = 'vehicle_name' if ranking_type == "Vehicle" else 'operator_name'
                
                # Use the comprehensive analyze function
                results_df, summary_stats, status_analytics = analyze_excel_file_comprehensive(
                    temp_file_path, group_by, mohh
                )
            
            # Rename column for display
            entity_col = f"{ranking_type.lower()}_name"
            results_df = results_df.rename(columns={group_by: entity_col})
            
            # Display summary cards
            st.markdown("### 📈 Performance Overview")
            create_summary_cards(summary_stats)
            
            # Create tabs for different views
            if show_option_analysis:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Performance Charts", "🔍 Option Distribution", "📋 Rankings Table", 
                    "🏆 Top Performers", "📉 Bottom Performers", "📊 Category Breakdown"
                ])
            else:
                tab1, tab3, tab4, tab5 = st.tabs([
                    "📊 Performance Charts", "📋 Rankings Table", 
                    "🏆 Top Performers", "📉 Bottom Performers"
                ])
            
            with tab1:
                st.markdown("### 📊 Performance Analytics")
                
                fig_availability, fig_hours, fig_scatter = create_performance_charts(results_df, ranking_type)
                
                st.plotly_chart(fig_availability, use_container_width=True)
                st.plotly_chart(fig_hours, use_container_width=True)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            if show_option_analysis:
                with tab2:
                    st.markdown("### 🔍 Option Distribution Analysis")
                    
                    # Get option distribution - read file again for original data
                    analyzer = VehicleMetricsAnalyzer(mohh=mohh)
                    original_df = pd.read_excel(uploaded_file)
                    option_distribution = analyzer.get_option_distribution(original_df, group_by)
                    
                    # Create charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_options = create_option_distribution_chart(option_distribution, top_n_options)
                        st.plotly_chart(fig_options, use_container_width=True)
                    
                    with col2:
                        fig_category = create_category_breakdown_chart(option_distribution)
                        st.plotly_chart(fig_category, use_container_width=True)
                    
                    # Display detailed option analytics
                    display_option_analytics(option_distribution, top_n_options)
                    
                    # Option summary table
                    st.markdown("### 📋 Option Summary Table")
                    option_summary = analyzer.get_option_summary_table(option_distribution)
                    st.dataframe(option_summary, use_container_width=True, height=400)
                    
                    # Download option analysis
                    csv_option = option_summary.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Option Analysis as CSV",
                        data=csv_option,
                        file_name="option_distribution_analysis.csv",
                        mime="text/csv"
                    )
                
                with tab6:
                    st.markdown("### 📊 Category vs Options Comparison")
                    
                    # Get category comparison
                    category_comparison = analyzer.get_category_vs_options_comparison(original_df)
                    
                    st.dataframe(category_comparison, use_container_width=True, height=500)
                    
                    # Download category comparison
                    csv_category = category_comparison.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Category Comparison as CSV",
                        data=csv_category,
                        file_name="category_vs_options_comparison.csv",
                        mime="text/csv"
                    )
            
            # Continue with other tabs...
            with tab3 if show_option_analysis else tab2:
                st.markdown(f"### 📋 Complete {ranking_type} Rankings")
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    min_pa = st.slider("Minimum PA% Filter", 0, 100, 0)
                with col2:
                    min_ua = st.slider("Minimum UA% Filter", 0, 100, 0)
                
                # Filter data
                filtered_df = results_df[
                    (results_df['PA(%)'] >= min_pa) & 
                    (results_df['UA(%)'] >= min_ua)
                ].reset_index(drop=True)
                
                if len(filtered_df) > 0:
                    st.dataframe(
                        style_dataframe(filtered_df),
                        use_container_width=True,
                        height=400
                    )
                    
                    st.success(f"Showing {len(filtered_df)} of {len(results_df)} {ranking_type.lower()}s")
                else:
                    st.warning("No data matches the current filters.")
            
            with tab4 if show_option_analysis else tab3:
                st.markdown(f"### 🏆 Top 10 {ranking_type}s")
                top_10 = results_df.head(10)
                
                for idx, row in top_10.iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            rank_emoji = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else f"#{row['rank']}"
                            st.write(f"**{rank_emoji} {row[entity_col]}**")
                        
                        with col2:
                            st.metric("PA%", f"{row['PA(%)']:.1f}%")
                        
                        with col3:
                            st.metric("UA%", f"{row['UA(%)']:.1f}%")
                        
                        with col4:
                            st.metric("Score", f"{row['availability_score']:.1f}")
                        
                        st.divider()
            
            with tab5 if show_option_analysis else tab4:
                st.markdown(f"### 📉 Bottom 10 {ranking_type}s")
                bottom_10 = results_df.tail(10)
                
                for idx, row in bottom_10.iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**#{row['rank']} {row[entity_col]}**")
                        
                        with col2:
                            pa_color = "status-danger" if row['PA(%)'] < pa_threshold else "status-warning" if row['PA(%)'] < 90 else "status-good"
                            st.markdown(f'<span class="{pa_color}">{row["PA(%)"]:.1f}%</span>', unsafe_allow_html=True)
                        
                        with col3:
                            ua_color = "status-danger" if row['UA(%)'] < ua_threshold else "status-warning" if row['UA(%)'] < 90 else "status-good"
                            st.markdown(f'<span class="{ua_color}">{row["UA(%)"]:.1f}%</span>', unsafe_allow_html=True)
                        
                        with col4:
                            st.write(f"{row['availability_score']:.1f}")
                        
                        # Show improvement suggestions
                        suggestions = []
                        if row['breakdown_hours'] > 20:
                            suggestions.append("🔧 Focus on preventive maintenance")
                        if row['delay_hours'] > 15:
                            suggestions.append("⏱️ Optimize scheduling and workflow")
                        if row['operation_hours'] < row['total_hours'] * 0.6:
                            suggestions.append("📈 Increase operational utilization")
                        
                        if suggestions:
                            st.caption(" | ".join(suggestions))
                        
                        st.divider()
            
            # Download section
            st.markdown("### 💾 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Rankings as CSV",
                    data=csv,
                    file_name=f"{ranking_type.lower()}_metrics_ranking.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download with multiple sheets
                excel_temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as excel_tmp_file:
                        excel_temp_path = excel_tmp_file.name
                        with pd.ExcelWriter(excel_temp_path, engine='xlsxwriter') as excel_buffer:
                            results_df.to_excel(excel_buffer, sheet_name=f'{ranking_type}_Rankings', index=False)
                            
                            # Add summary sheet
                            summary_df = pd.DataFrame([summary_stats])
                            summary_df.to_excel(excel_buffer, sheet_name='Summary', index=False)
                            
                            # Add option analysis if enabled
                            if show_option_analysis:
                                option_summary = analyzer.get_option_summary_table(option_distribution)
                                option_summary.to_excel(excel_buffer, sheet_name='Option_Analysis', index=False)
                    
                    with open(excel_temp_path, 'rb') as f:
                        st.download_button(
                            label="📊 Download Complete Analysis",
                            data=f.read(),
                            file_name=f"{ranking_type.lower()}_complete_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                finally:
                    # Clean up Excel temp file
                    if excel_temp_path:
                        safe_remove_temp_file(excel_temp_path)
            
            with col3:
                # Status analytics as JSON
                if show_option_analysis:
                    import json
                    json_data = json.dumps(status_analytics, indent=2, default=str)
                    st.download_button(
                        label="📋 Download Status Analytics",
                        data=json_data,
                        file_name="status_analytics.json",
                        mime="application/json"
                    )
            
        except FileNotFoundError as e:
            st.error(f"📁 File not found: {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)  # Show detailed error for debugging
        
        finally:
            # Clean up temp file in finally block
            if temp_file_path:
                safe_remove_temp_file(temp_file_path)
    
    else:
        # Welcome message
        st.markdown("""
        <div class="upload-section">
            <h3>🎯 Welcome to Vehicle & Operator Analytics!</h3>
            <p>Upload your Excel file containing vehicle/operator status data to get started with comprehensive performance analysis.</p>
            <p><strong>Required columns:</strong> vehicle_name, operator_name, status, duration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data format
        with st.expander("📋 Expected Data Format"):
            sample_data = pd.DataFrame({
                'vehicle_name': ['Truck_001', 'Truck_002', 'Truck_001'],
                'operator_name': ['John_Doe', 'Jane_Smith', 'John_Doe'],
                'status': ['READY', 'BREAKDOWN', 'DELAY'],
                'duration': ['2:30:00', '1:15:30', '0:45:00']
            })
            st.dataframe(sample_data, use_container_width=True)
            
        # Coal hauling status options
        with st.expander("🏭 Coal Hauling Status Options"):
            st.markdown("""
            **READY Status Options:**
            - Coal Hauling
            
            **IDLE Status Options:**
            - Hujan (Rain)
            - Demo
            
            **DELAY Status Options:**
            - Antrian Jembatan Timbang, Fatigue Test, Internal Problem
            - Jembatan Timbang Bermasalah, Jetty Crowded, Makan/Istirahat
            - P2H, P5M, Pembersihan Unit, Pengisian Bahan Bakar
            - And 10+ more operational delay categories
            
            **BREAKDOWN Status Options:**
            - Periodic Inspection, Schedule Maintenance
            - Tire Maintenance, Unschedule Maintenance
            """)

if __name__ == "__main__":
    main()