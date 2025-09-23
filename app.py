import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from analyze import analyze_excel_file

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
</style>
""", unsafe_allow_html=True)

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
        
        # About section
        with st.expander("ℹ️ About This Tool"):
            st.write("""
            **Physical Availability (PA%)**: Percentage of time equipment is not broken down
            
            **Utilization Availability (UA%)**: Percentage of available time actually used
            
            **Efficiency (%)**: Percentage of total logged time spent in operation
            
            **Availability Score**: Average of PA% and UA%
            """)
    
    # File upload section
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Upload your vehicle/operator status data file"
    )
    
    if uploaded_file is not None:
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
            
            # Save uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process data
            with st.spinner('🔄 Processing data...'):
                group_by = 'vehicle_name' if ranking_type == "Vehicle" else 'operator_name'
                
                # Use the enhanced analyze_excel_file function
                results_df, summary_stats = analyze_excel_file(temp_file_path, group_by, mohh)
                
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            
            # Rename column for display
            entity_col = f"{ranking_type.lower()}_name"
            results_df = results_df.rename(columns={group_by: entity_col})
            
            # Display summary cards
            st.markdown("### 📈 Performance Overview")
            create_summary_cards(summary_stats)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📋 Rankings Table", "🏆 Top Performers", "📉 Bottom Performers"])
            
            with tab1:
                st.markdown("### 📊 Performance Analytics")
                
                fig_availability, fig_hours, fig_scatter = create_performance_charts(results_df, ranking_type)
                
                st.plotly_chart(fig_availability, use_container_width=True)
                st.plotly_chart(fig_hours, use_container_width=True)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab2:
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
            
            with tab3:
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
            
            with tab4:
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
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"{ranking_type.lower()}_metrics_ranking.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download
                excel_buffer = pd.ExcelWriter('temp_output.xlsx', engine='xlsxwriter')
                results_df.to_excel(excel_buffer, sheet_name=f'{ranking_type}_Rankings', index=False)
                
                # Add summary sheet
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_excel(excel_buffer, sheet_name='Summary', index=False)
                
                excel_buffer.close()
                
                with open('temp_output.xlsx', 'rb') as f:
                    st.download_button(
                        label="📊 Download as Excel",
                        data=f.read(),
                        file_name=f"{ranking_type.lower()}_metrics_ranking.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Clean up temp file
                if os.path.exists('temp_output.xlsx'):
                    os.remove('temp_output.xlsx')
            
        except FileNotFoundError as e:
            st.error(f"📁 File not found: {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)  # Show detailed error for debugging
    
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
                'status': ['OPERATION', 'BREAKDOWN', 'DELAY'],
                'duration': ['2:30:00', '1:15:30', '0:45:00']
            })
            st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()