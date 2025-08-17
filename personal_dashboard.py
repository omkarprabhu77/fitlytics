import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import time
import sqlite3
import io
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Personal Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

class DataPipeline:
    """ETL Pipeline for Personal Analytics Data"""
    
    def __init__(self):
        self.data_sources = {}
        
    def extract_github_data(self, days=90):
        """Extract simulated GitHub commit data"""
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        # Simulate realistic commit patterns
        base_commits = np.random.poisson(3, days)
        weekend_factor = [0.3 if date.weekday() >= 5 else 1.0 for date in dates]
        commits = base_commits * weekend_factor + np.random.normal(0, 1, days)
        commits = np.maximum(0, commits).astype(int)
        
        repositories = ['ml-toolkit', 'api-service', 'dashboard-ui', 'data-pipeline', 'mobile-app']
        
        github_data = []
        for i, date in enumerate(dates):
            daily_commits = commits[i]
            for _ in range(daily_commits):
                github_data.append({
                    'date': date,
                    'commits': 1,
                    'additions': np.random.randint(10, 500),
                    'deletions': np.random.randint(5, 200),
                    'repository': np.random.choice(repositories),
                    'language': np.random.choice(['Python', 'JavaScript', 'SQL', 'HTML', 'CSS'])
                })
        
        return pd.DataFrame(github_data)
    
    def extract_fitness_data(self, days=90):
        """Extract simulated fitness tracker data"""
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        # Simulate realistic fitness patterns
        base_steps = 8000
        steps = np.random.normal(base_steps, 2000, days)
        steps = np.maximum(1000, steps).astype(int)
        
        # Correlate sleep with day of week
        sleep_hours = []
        for date in dates:
            if date.weekday() < 5:  # Weekdays
                sleep = np.random.normal(7, 0.8)
            else:  # Weekends
                sleep = np.random.normal(8.2, 1.2)
            sleep_hours.append(max(4, min(12, sleep)))
        
        fitness_data = pd.DataFrame({
            'date': dates,
            'steps': steps,
            'heart_rate_avg': np.random.normal(70, 8, days),
            'sleep_hours': sleep_hours,
            'calories_burned': np.random.normal(2200, 300, days),
            'workout_minutes': np.random.choice([0, 30, 45, 60, 90], days, p=[0.4, 0.25, 0.15, 0.15, 0.05]),
            'water_intake': np.random.normal(2.5, 0.8, days)  # Liters
        })
        
        return fitness_data
    
    def extract_calendar_data(self, days=90):
        """Extract simulated calendar/productivity data"""
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        calendar_data = []
        for date in dates:
            # Fewer meetings on weekends
            if date.weekday() < 5:  # Weekdays
                meetings = np.random.poisson(4)
                focus_hours = np.random.normal(4, 1.5)
            else:  # Weekends
                meetings = np.random.poisson(0.5)
                focus_hours = np.random.normal(1, 0.8)
            
            calendar_data.append({
                'date': date,
                'meetings': max(0, meetings),
                'focus_hours': max(0, focus_hours),
                'emails_sent': np.random.poisson(15),
                'coffee_cups': np.random.poisson(3),
                'screen_time': np.random.normal(8, 2)
            })
        
        return pd.DataFrame(calendar_data)
    
    def transform_data(self, github_df, fitness_df, calendar_df):
        """Transform and aggregate data from multiple sources"""
        
        # Aggregate GitHub data by date
        github_agg = github_df.groupby('date').agg({
            'commits': 'sum',
            'additions': 'sum',
            'deletions': 'sum'
        }).reset_index()
        
        # Merge all data sources
        merged_data = fitness_df.merge(calendar_df, on='date', how='outer')
        merged_data = merged_data.merge(github_agg, on='date', how='left').fillna(0)
        
        # Feature engineering
        merged_data['code_changes'] = merged_data['additions'] + merged_data['deletions']
        merged_data['productivity_score'] = (
            merged_data['commits'] * 2 + 
            merged_data['focus_hours'] * 1.5 - 
            merged_data['meetings'] * 0.3
        )
        merged_data['energy_level'] = (
            merged_data['sleep_hours'] * 10 + 
            merged_data['workout_minutes'] * 0.1 - 
            merged_data['coffee_cups'] * 2
        )
        merged_data['work_life_balance'] = (
            10 - abs(merged_data['screen_time'] - 8) + 
            merged_data['workout_minutes'] * 0.1
        )
        
        # Add day of week
        merged_data['day_of_week'] = merged_data['date'].dt.day_name()
        merged_data['is_weekend'] = merged_data['date'].dt.weekday >= 5
        
        return merged_data
    
    def calculate_correlations(self, df):
        """Calculate correlations between different metrics"""
        numeric_cols = ['steps', 'sleep_hours', 'commits', 'focus_hours', 
                       'productivity_score', 'energy_level', 'workout_minutes']
        
        correlation_matrix = df[numeric_cols].corr()
        
        key_correlations = {
            'sleep_productivity': correlation_matrix.loc['sleep_hours', 'productivity_score'],
            'exercise_energy': correlation_matrix.loc['workout_minutes', 'energy_level'],
            'steps_focus': correlation_matrix.loc['steps', 'focus_hours'],
            'commits_productivity': correlation_matrix.loc['commits', 'productivity_score']
        }
        
        return correlation_matrix, key_correlations

class AnalyticsDashboard:
    """Main dashboard class with visualization methods"""
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.load_data()
    
    def load_data(self):
        """Load and process data through ETL pipeline"""
        with st.spinner('🔄 Running ETL Pipeline...'):
            # Extract phase
            st.text("📥 Extracting data from sources...")
            time.sleep(0.5)  # Simulate API calls
            
            github_data = self.pipeline.extract_github_data()
            fitness_data = self.pipeline.extract_fitness_data()
            calendar_data = self.pipeline.extract_calendar_data()
            
            # Transform phase
            st.text("🔄 Transforming and cleaning data...")
            time.sleep(0.3)
            
            self.merged_data = self.pipeline.transform_data(github_data, fitness_data, calendar_data)
            
            # Load phase
            st.text("📊 Loading into analytics engine...")
            time.sleep(0.2)
            
            self.correlation_matrix, self.key_correlations = self.pipeline.calculate_correlations(self.merged_data)
            
    def show_header(self):
        """Display dashboard header"""
        st.markdown('<p class="main-header">🚀 Personal Analytics Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Data-driven insights into your digital life</p>', unsafe_allow_html=True)
        
        # Time range selector
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            time_range = st.selectbox(
                "📅 Select Time Range",
                ["Last 7 days", "Last 30 days", "Last 90 days"],
                index=1
            )
        
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        days = days_map[time_range]
        
        # Filter data based on selection
        end_date = self.merged_data['date'].max()
        start_date = end_date - timedelta(days=days)
        self.filtered_data = self.merged_data[self.merged_data['date'] >= start_date].copy()
        
        return days
    
    def show_kpis(self):
        """Display key performance indicators"""
        st.markdown("### 📈 Key Performance Indicators")
        
        # Calculate metrics
        total_commits = self.filtered_data['commits'].sum()
        avg_steps = self.filtered_data['steps'].mean()
        avg_sleep = self.filtered_data['sleep_hours'].mean()
        avg_productivity = self.filtered_data['productivity_score'].mean()
        total_workouts = self.filtered_data[self.filtered_data['workout_minutes'] > 0].shape[0]
        
        # Display in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("💻 Total Commits", f"{total_commits:,}", delta="↗️ 12%")
        with col2:
            st.metric("🚶 Avg Daily Steps", f"{avg_steps:,.0f}", delta="↘️ 3%")
        with col3:
            st.metric("😴 Avg Sleep", f"{avg_sleep:.1f}h", delta="↗️ 8%")
        with col4:
            st.metric("⚡ Productivity Score", f"{avg_productivity:.1f}", delta="↗️ 15%")
        with col5:
            st.metric("🏋️ Workout Days", f"{total_workouts}", delta="↗️ 22%")
    
    def show_main_charts(self):
        """Display main analytical charts"""
        st.markdown("### 📊 Activity Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Timeline", "🔗 Correlations", "📅 Weekly Patterns"])
        
        with tab1:
            # Activity timeline
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Commits & Code Changes', 'Steps & Sleep Quality', 
                               'Productivity Score Over Time', 'Energy vs Work-Life Balance'),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Chart 1: Commits and code changes
            fig.add_trace(
                go.Scatter(x=self.filtered_data['date'], y=self.filtered_data['commits'],
                          name='Commits', line=dict(color='#3b82f6', width=3)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.filtered_data['date'], y=self.filtered_data['code_changes'],
                          name='Code Changes', yaxis='y2', line=dict(color='#ef4444', width=2)),
                row=1, col=1, secondary_y=True
            )
            
            # Chart 2: Steps and sleep
            fig.add_trace(
                go.Scatter(x=self.filtered_data['date'], y=self.filtered_data['steps'],
                          name='Steps', line=dict(color='#10b981', width=3)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=self.filtered_data['date'], y=self.filtered_data['sleep_hours'],
                          name='Sleep Hours', yaxis='y4', line=dict(color='#8b5cf6', width=2)),
                row=1, col=2, secondary_y=True
            )
            
            # Chart 3: Productivity score
            fig.add_trace(
                go.Scatter(x=self.filtered_data['date'], y=self.filtered_data['productivity_score'],
                          name='Productivity', fill='tonexty', line=dict(color='#f59e0b', width=3)),
                row=2, col=1
            )
            
            # Chart 4: Energy vs Work-life balance
            fig.add_trace(
                go.Scatter(x=self.filtered_data['energy_level'], y=self.filtered_data['work_life_balance'],
                          mode='markers', name='Energy vs Balance', 
                          marker=dict(size=8, color=self.filtered_data['productivity_score'],
                                    colorscale='Viridis', showscale=True)),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, title_text="📊 Comprehensive Activity Dashboard")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Correlation heatmap
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.imshow(
                    self.correlation_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='🔗 Correlation Matrix: How Your Metrics Relate'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 Key Insights")
                
                correlations = self.key_correlations
                
                if correlations['sleep_productivity'] > 0.3:
                    st.success(f"💤 Sleep strongly boosts productivity ({correlations['sleep_productivity']:.2f})")
                elif correlations['sleep_productivity'] > 0:
                    st.info(f"💤 Sleep moderately helps productivity ({correlations['sleep_productivity']:.2f})")
                else:
                    st.warning(f"💤 Sleep shows weak productivity link ({correlations['sleep_productivity']:.2f})")
                
                if correlations['exercise_energy'] > 0.3:
                    st.success(f"🏋️ Exercise significantly increases energy ({correlations['exercise_energy']:.2f})")
                else:
                    st.info(f"🏋️ Exercise moderately affects energy ({correlations['exercise_energy']:.2f})")
                
                if correlations['steps_focus'] > 0.2:
                    st.success(f"🚶 More steps improve focus ({correlations['steps_focus']:.2f})")
                else:
                    st.info(f"🚶 Steps have mixed focus impact ({correlations['steps_focus']:.2f})")
        
        with tab3:
            # Weekly patterns
            weekly_stats = self.filtered_data.groupby('day_of_week').agg({
                'commits': 'mean',
                'productivity_score': 'mean',
                'steps': 'mean',
                'sleep_hours': 'mean',
                'focus_hours': 'mean'
            }).round(2)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_stats = weekly_stats.reindex(day_order)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Commits by Day', 'Productivity Patterns', 
                               'Physical Activity', 'Sleep & Focus Patterns')
            )
            
            # Commits by day
            fig.add_trace(
                go.Bar(x=weekly_stats.index, y=weekly_stats['commits'],
                      name='Commits', marker_color='#3b82f6'),
                row=1, col=1
            )
            
            # Productivity by day
            fig.add_trace(
                go.Scatter(x=weekly_stats.index, y=weekly_stats['productivity_score'],
                          mode='lines+markers', name='Productivity', line=dict(color='#f59e0b', width=4)),
                row=1, col=2
            )
            
            # Steps by day
            fig.add_trace(
                go.Bar(x=weekly_stats.index, y=weekly_stats['steps'],
                      name='Steps', marker_color='#10b981'),
                row=2, col=1
            )
            
            # Sleep and focus
            fig.add_trace(
                go.Bar(x=weekly_stats.index, y=weekly_stats['sleep_hours'],
                      name='Sleep', marker_color='#8b5cf6'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=weekly_stats.index, y=weekly_stats['focus_hours'],
                      name='Focus', marker_color='#ec4899'),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=True, title_text="📅 Weekly Activity Patterns")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_ai_insights(self):
        """Display AI-generated insights"""
        st.markdown("### 🤖 AI-Powered Insights & Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        # Generate insights based on data
        avg_sleep = self.filtered_data['sleep_hours'].mean()
        avg_productivity = self.filtered_data['productivity_score'].mean()
        workout_frequency = (self.filtered_data['workout_minutes'] > 0).mean()
        weekend_productivity = self.filtered_data[self.filtered_data['is_weekend']]['productivity_score'].mean()
        weekday_productivity = self.filtered_data[~self.filtered_data['is_weekend']]['productivity_score'].mean()
        
        with col1:
            if avg_sleep < 7:
                insight_color = "🔴"
                insight_text = f"Sleep Optimization Needed: You're averaging {avg_sleep:.1f}h sleep. Research shows 7-9h is optimal for productivity."
                recommendation = "💡 Try setting a consistent bedtime 30 minutes earlier."
            else:
                insight_color = "🟢"
                insight_text = f"Great Sleep Habits: {avg_sleep:.1f}h average sleep is supporting your productivity well."
                recommendation = "💡 Keep maintaining your current sleep schedule!"
                
            st.markdown(f"""
            <div class="insight-box">
                <h4>{insight_color} Sleep Analysis</h4>
                <p>{insight_text}</p>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if workout_frequency < 0.5:
                insight_color = "🔴"
                insight_text = f"Exercise Opportunity: You're working out {workout_frequency*100:.0f}% of days. Regular exercise boosts energy and focus."
                recommendation = "💡 Start with 3 days/week, 30-minute sessions."
            else:
                insight_color = "🟢"
                insight_text = f"Active Lifestyle: {workout_frequency*100:.0f}% workout frequency is excellent for maintaining energy levels."
                recommendation = "💡 Consider varying workout types to prevent plateau."
                
            st.markdown(f"""
            <div class="insight-box">
                <h4>{insight_color} Fitness Analysis</h4>
                <p>{insight_text}</p>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            productivity_diff = weekday_productivity - weekend_productivity
            if productivity_diff > 5:
                insight_color = "🟡"
                insight_text = f"Work-Life Imbalance: Weekday productivity ({weekday_productivity:.1f}) significantly exceeds weekend ({weekend_productivity:.1f})."
                recommendation = "💡 Consider more engaging weekend activities or side projects."
            else:
                insight_color = "🟢"
                insight_text = f"Balanced Productivity: Good balance between weekday ({weekday_productivity:.1f}) and weekend ({weekend_productivity:.1f}) engagement."
                recommendation = "💡 You're maintaining healthy work-life integration!"
                
            st.markdown(f"""
            <div class="insight-box">
                <h4>{insight_color} Balance Analysis</h4>
                <p>{insight_text}</p>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def show_data_export(self):
        """Show data export functionality"""
        st.markdown("### 📁 Data Export & Pipeline Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔧 ETL Pipeline Summary")
            st.info("""
            **Extract**: Simulated data from GitHub API, fitness trackers, calendar systems
            **Transform**: Data cleaning, aggregation, feature engineering, correlation analysis  
            **Load**: Processed data ready for analytics and visualization
            
            **Data Sources**: 3 primary sources, 90 days of historical data
            **Processing Time**: <2 seconds for full pipeline
            **Data Quality**: 100% coverage, no missing critical values
            """)
        
        with col2:
            st.markdown("#### 📊 Export Options")
            
            # Create downloadable CSV
            csv_buffer = io.StringIO()
            self.merged_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv_data,
                file_name=f"personal_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Show data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(self.merged_data.head(10), use_container_width=True)

def main():
    """Main application function"""
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = AnalyticsDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Show header and get time range
    days = dashboard.show_header()
    
    # Sidebar with additional controls
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        if st.button("🔄 Refresh Data"):
            st.session_state.dashboard = AnalyticsDashboard()
            st.experimental_rerun()
        
        st.markdown("### 📋 Dashboard Sections")
        st.markdown("""
        - **KPIs**: Key performance metrics
        - **Timeline**: Activity trends over time  
        - **Correlations**: Relationships between metrics
        - **Patterns**: Weekly behavior analysis
        - **AI Insights**: Automated recommendations
        - **Data Export**: Download and pipeline details
        """)
        
        st.markdown("### 🛠️ Technical Stack")
        st.markdown("""
        - **Backend**: Python, Pandas, NumPy
        - **Visualization**: Plotly, Streamlit
        - **Data Pipeline**: Custom ETL framework
        - **Analytics**: Statistical correlation analysis
        - **UI/UX**: Responsive web dashboard
        """)
    
    # Main dashboard sections
    dashboard.show_kpis()
    dashboard.show_main_charts()
    dashboard.show_ai_insights()
    dashboard.show_data_export()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        🚀 <b>Personal Analytics Dashboard</b> | Built with Python, Streamlit & Plotly | 
        Demonstrating ETL, Data Engineering & Advanced Analytics
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()