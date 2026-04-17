import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from plotly.graph_objects import Figure
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from portkey_ai import PORTKEY_GATEWAY_URL
import httpx
import re
import time


st.set_page_config(page_title="Data Analysis & Visualization Tool", layout="wide")
st.title("Data Analysis & Visualization Tool")



st.markdown("""
    <style>
    /* Light mode styles */
    @media (prefers-color-scheme: light) {      
        .stButton > button {
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(0);
            background: linear-gradient(145deg, #ffffff, #e6e6e6);
            color: #333333;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid rgba(0, 0, 0, 0.1);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
            background: linear-gradient(145deg, #ffffff, #e6e6e6);
            color: #333333;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(0);
        }

        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(145deg, #e6e6e6, #ffffff);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transform: translateY(1px);
        }
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .stButton > button {
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            box-shadow: 0 4px 6px rgba(98, 81, 81, 0.31);
            transform: translateY(0);
            background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(98, 81, 81, 0.78);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
            background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(98, 81, 81, 0.31);
            transform: translateY(0);
        }

        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(98, 81, 81, 0.78);
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
            box-shadow: 0 2px 4px rgba(98, 81, 81, 0.31);
            transform: translateY(1px);
        }
    }

    /* Shared click effect */
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* File uploader styling */
    .stFileUploader > button {
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
        position: relative;
        transform: translateY(0);
    }

    @media (prefers-color-scheme: light) {
        .stFileUploader > button {
            background: linear-gradient(145deg, #ffffff, #e6e6e6);
            color: #333333;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    }

    @media (prefers-color-scheme: dark) {
        .stFileUploader > button {
            background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(98, 81, 81, 0.31);
        }
    }

    .stFileUploader > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }

    .stFileUploader > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)



def get_llm_client(temperature=0.5):
    """Create and return a LangChain ChatOpenAI client using st.secrets"""
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        portkey_api_key = st.secrets["PORTKEY_API_KEY"]

        if not openai_api_key or not portkey_api_key:
            st.error("API keys are missing. Please check your Streamlit secrets.")
            return None
        
        headers = {
            "X-Portkey-Provider": "openai",
            "X-Portkey-Api-Key": portkey_api_key
        }
        
        http_client = httpx.Client(
            base_url=PORTKEY_GATEWAY_URL,
            headers=headers
        )
        
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=openai_api_key,
            http_client=http_client,
            base_url=PORTKEY_GATEWAY_URL
        )
    except Exception as e:
        st.error(f"Error initializing LangChain client: {str(e)}")
        return None

def load_data():
    """Load and clean data from uploaded file"""
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read and clean the CSV file
            df = pd.read_csv(uploaded_file)
            df = clean_column_names(df)
            
            # Create SQLite database with cleaned column names
            st.session_state.db_conn = csv_to_sqlite(df)
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    return None

def clean_column_names(df):
    """
    Clean DataFrame column names by:
    1. Removing leading and trailing spaces
    2. Replacing multiple spaces with single space
    """
    # Clean all column names
    df.columns = [
        re.sub(r'\s+', ' ', col.strip()) 
        for col in df.columns
    ]
    
    return df

def create_filters(df):
    """Create and return filter selections"""
    with st.sidebar:
        st.header("Data Filters")
        filters = {}
        
        # Define filter columns with their types
        filter_config = {
            'numeric': ['Age'],
            'categorical': [
                'Employee Code', 'Reporting Manager', 'Branch / Location',
                'Gender', 'Business Unit', 'Department', 'Designation'
            ]
        }
        
        # Create numeric filters
        for col in filter_config['numeric']:
            if col in df.columns:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                filters[col] = st.slider(
                    f"Filter by {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
        
        # Create categorical filters
        for col in filter_config['categorical']:
            if col in df.columns:
                unique_values = sorted(df[col].dropna().unique().tolist())
                filters[col] = st.multiselect(
                    f"Filter by {col}",
                    options=unique_values,
                    default=[],
                    max_selections=5
                )
        
        return filters

def apply_filters(df, filters):
    """Apply selected filters to DataFrame"""
    filtered_df = df.copy()
    for col, values in filters.items():
        if isinstance(values, list) and values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        elif isinstance(values, tuple):
            filtered_df = filtered_df[
                (filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])
            ]
    return filtered_df

def create_visualization_from_query(df: pd.DataFrame, query_results: pd.DataFrame) -> Figure:
    """Create appropriate visualizations based on query results"""
    if query_results is None or query_results.empty:
        return None
        
    numeric_columns = query_results.select_dtypes(include=['number']).columns
    categorical_columns = query_results.select_dtypes(include=['object']).columns
    
    figures = []
    
    # Create bar chart for categorical data
    if len(categorical_columns) >= 1:
        if len(numeric_columns) >= 1:
            # Bar chart with numeric values
            fig = px.bar(
                query_results,
                x=categorical_columns[0],
                y=numeric_columns[0],
                title=f"{numeric_columns[0]} by {categorical_columns[0]}"
            )
        else:
            # Count-based bar chart
            counts = query_results[categorical_columns[0]].value_counts()
            fig = px.bar(
                counts,
                title=f"Distribution of {categorical_columns[0]}"
            )
        figures.append(fig)
    
    # Create pie chart for categorical data
    if len(categorical_columns) >= 1:
        counts = query_results[categorical_columns[0]].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Distribution of {categorical_columns[0]}"
        )
        figures.append(fig)
    
    # Create line or scatter plot for numeric data
    if len(numeric_columns) >= 2:
        fig = px.scatter(
            query_results,
            x=numeric_columns[0],
            y=numeric_columns[1],
            title=f"{numeric_columns[1]} vs {numeric_columns[0]}"
        )
        figures.append(fig)
    
    return figures

def natural_language_query_section(df, llm):
    """Handle natural language query section with column selection and select all functionality"""
    # Initialize session state
    init_query_session_state(None)
    
    st.header("📝 Natural Language Query")
    
    # Get all available columns across all tables with category prefixes for unique keys
    all_columns = {
        'Core Info': [
            'Employee Code', 'Employee Name', 'Birth Date', 
            'Gender', 'Ethnicity', 'Marital status', 'Citizenship',
            'Age', 'Age Range'
        ],
        'Education': [
            'Class 10 Percentage', 'Class 12 Percentage',
            'Highest Education Level', 'College / Institution - UG',
            'UG College affiliated to', 'UG Degree', 'UG CGPA', 'UG YOP',
            'PG/Diploma Course Institution/University', 'PG/Diploma Course Specialization',
            'PG/Diploma Course CGPA', 'PG YOP'
        ],
        'Employment': [
            'Legal Entity', 'Branch / Location', 'Business Unit',
            'Department', 'Designation', 'Employment type', 'Grade',
            'Reporting Manager', 'Facility - Seat Number / Remote',
            'Joining Date', 'Last Working Date', 'Resignation Date',
            'Status of employment(Active / Inactive / Suspended)',
            'Overall Work Experience(in Years)', 'Tenure(in years)', 'Tenure Range'
        ],
        'Payroll': [
            'Payroll Month', 'Payroll currency',
            'Monthly Basic', 'Monthly House Rent Allowance', 'Dearness Allowance',
            'Monthly Allowance', 'Monthly Special bonus', 'Monthly Hardship Allowance',
            'Monthly Net Compensation', 'Monthly Gross Compensation',
            'Monthly CTC'
        ],
        'Performance': [
            '2022 Performance ratings', '2022 Feedback from performance reviews',
            '2023 Performance ratings', '2023 Feedback from performance reviews',
            'Key performance indicators (KPIs)', 'Attendance records',
            'Availed Casual Leave', 'Availed Sick Leave', 'Availed Vacation Leave'
        ]
    }
    
    # Initialize session state for checkboxes if not exists
    if 'select_all_global' not in st.session_state:
        st.session_state.select_all_global = False
    
    if 'select_all_states' not in st.session_state:
        st.session_state.select_all_states = {category: False for category in all_columns.keys()}
    
    if 'column_states' not in st.session_state:
        st.session_state.column_states = {}
        for category, cols in all_columns.items():
            category_prefix = category.lower().replace(' ', '_')
            for col in cols:
                checkbox_key = f"{category_prefix}_col_{col.lower().replace(' ', '_')}"
                st.session_state.column_states[checkbox_key] = False
    
    # Column selection
    selected_columns = []
    with st.expander("Select Columns to Display", expanded=False):
        # Function to update all checkbox states
        def update_all_checkboxes(value):
            st.session_state.select_all_global = value
            for category in all_columns.keys():
                st.session_state.select_all_states[category] = value
                category_prefix = category.lower().replace(' ', '_')
                for col in all_columns[category]:
                    checkbox_key = f"{category_prefix}_col_{col.lower().replace(' ', '_')}"
                    st.session_state.column_states[checkbox_key] = value
        
        # Global select all checkbox
        if st.checkbox("Select All Columns", key="global_select_all", value=st.session_state.select_all_global):
            update_all_checkboxes(True)
        else:
            if st.session_state.select_all_global:  # Only update if it was previously selected
                update_all_checkboxes(False)
        
        st.markdown("---")
        
        # Create columns for better organization
        for category, cols in all_columns.items():
            st.subheader(category)
            category_prefix = category.lower().replace(' ', '_')
            
            # Function to handle category select all
            def handle_category_select_all(category_name, prefix, columns):
                select_all_key = f"select_all_{prefix}"
                if st.checkbox(
                    f"Select All {category_name}",
                    key=select_all_key,
                    value=st.session_state.select_all_states[category_name]
                ):
                    # Update all columns in this category
                    for col in columns:
                        checkbox_key = f"{prefix}_col_{col.lower().replace(' ', '_')}"
                        st.session_state.column_states[checkbox_key] = True
                    st.session_state.select_all_states[category_name] = True
                else:
                    # Only update if it was previously selected
                    if st.session_state.select_all_states[category_name]:
                        for col in columns:
                            checkbox_key = f"{prefix}_col_{col.lower().replace(' ', '_')}"
                            st.session_state.column_states[checkbox_key] = False
                        st.session_state.select_all_states[category_name] = False
            
            # Add select all checkbox for category
            handle_category_select_all(category, category_prefix, cols)
            
            # Add individual column checkboxes
            cols_selected = []
            for col in cols:
                checkbox_key = f"{category_prefix}_col_{col.lower().replace(' ', '_')}"
                
                is_checked = st.checkbox(
                    col,
                    value=st.session_state.column_states[checkbox_key],
                    key=checkbox_key
                )
                
                # Update session state
                st.session_state.column_states[checkbox_key] = is_checked
                
                if is_checked:
                    cols_selected.append(col)
            
            # Update selected columns
            selected_columns.extend(cols_selected)
            
            # Update category select all based on individual selections
            all_selected = all(st.session_state.column_states[f"{category_prefix}_col_{col.lower().replace(' ', '_')}"] 
                             for col in cols)
            st.session_state.select_all_states[category] = all_selected
            
            # Update global select all based on all categories
            all_categories_selected = all(st.session_state.select_all_states.values())
            st.session_state.select_all_global = all_categories_selected
            
            st.markdown("---")

    # Rest of the function remains the same...
    natural_query = st.text_area(
        "Enter your question in plain English:",
        placeholder="Example: Show me all employees who are full-time and their salaries",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    if col1.button("Generate Visualizations"):
        if natural_query and st.session_state.db_conn and llm:
            with st.spinner("Generating visualizations..."):
                modified_query = modify_query_for_selected_columns(natural_query, selected_columns)
                table_schema = get_table_schema(df)
                sql_query = convert_to_sql(llm, modified_query, table_schema)
                
                if sql_query:
                    result_df = execute_fixed_sql_query(st.session_state.db_conn, sql_query)
                    if result_df is not None:
                        st.session_state['query_result_df'] = result_df
                        st.session_state['show_viz'] = True
                        st.session_state['current_query'] = natural_query
    
    if col2.button("Generate AI Analysis"):
        if natural_query and st.session_state.db_conn and llm:
            with st.spinner("Generating AI analysis..."):
                modified_query = modify_query_for_selected_columns(natural_query, selected_columns)
                table_schema = get_table_schema(df)
                sql_query = convert_to_sql(llm, modified_query, table_schema)
                
                if sql_query:
                    result_df = execute_fixed_sql_query(st.session_state.db_conn, sql_query)
                    if result_df is not None:
                        st.session_state['query_result_df'] = result_df
                        st.session_state['show_viz'] = True
                        st.session_state['current_query'] = natural_query
                        
                        st.subheader("AI Insights")
                        insights = generate_insights(llm, result_df, natural_query, sql_query)
                        st.write(insights)
    
    # Display query results if they exist
    if st.session_state['query_result_df'] is not None:
        st.subheader(f"Query Results for: {st.session_state.get('current_query', 'Previous Query')}")
        st.dataframe(st.session_state['query_result_df'])
        
        if st.session_state['show_viz']:
            show_visualization_options(st.session_state['query_result_df'])

def modify_query_for_selected_columns(natural_query, selected_columns):
    """Modify the natural query to include only selected columns"""
    if not selected_columns:
        # If no columns selected, return original query
        return natural_query
    
    # Always include Employee Code
    if 'Employee Code' not in selected_columns:
        selected_columns.append('Employee Code')
    
    # Add column selection to the query
    columns_str = ', '.join([f'"{col}"' for col in selected_columns])
    
    # Check if the query involves payroll/salary data
    involves_payroll = any(term in natural_query.lower() 
                         for term in ['salary', 'payroll', 'pay', 'compensation', 'monthly'])
    
    # Add Payroll Month if it's a payroll query
    if involves_payroll and 'Payroll Month' not in selected_columns:
        columns_str = '"Payroll Month", ' + columns_str
    
    modified_query = f"{natural_query} showing only these columns: {columns_str}"
    return modified_query

def create_default_visualizations(df):
    """Create default visualizations with filter support"""
    import time
    default_graphs = []
    
    # 1. Department Distribution
    default_graphs.append({
        'title': 'Department Distribution',
        'type': 'Pie Chart',
        'id': f"default_1_{int(time.time() * 1000)}",
        'config': {
            'type': 'Pie Chart',
            'x_axis': 'Department',
            'y_axis': None
        }
    })
    
    # 2. Age Distribution
    default_graphs.append({
        'title': 'Age Distribution',
        'type': 'Histogram',
        'id': f"default_2_{int(time.time() * 1000)}",
        'config': {
            'type': 'Histogram',
            'x_axis': 'Age',
            'y_axis': None
        }
    })
    
    # 3. Gender Distribution by Department
    default_graphs.append({
        'title': 'Gender Distribution by Department',
        'type': 'Bar Chart',
        'id': f"default_3_{int(time.time() * 1000)}",
        'config': {
            'type': 'Bar Chart',
            'x_axis': 'Department',
            'y_axis': 'count'
        }
    })
    
    return default_graphs

def visualization_section(df):
    """Handle visualization section with filter support"""
    st.header("📊 Data Visualization")
    
    # Initialize saved_graphs in session state if not exists
    if 'saved_graphs' not in st.session_state:
        st.session_state.saved_graphs = create_default_visualizations(df)
    else:
        # Ensure all existing graphs have IDs and configuration
        for i, graph in enumerate(st.session_state.saved_graphs):
            if 'id' not in graph:
                graph['id'] = str(i)
    
    # Create tabs for different visualization options
    tab1, tab2 = st.tabs(["Create New Visualization", "Saved Visualizations"])
    
    with tab1:
        graph_type = st.selectbox(
            "Select Graph Type",
            ["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot", "Histogram"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col2:
            if graph_type not in ["Pie Chart", "Histogram"]:
                y_axis = st.selectbox(
                    "Select Y-axis",
                    df.select_dtypes(include=['number']).columns
                )
            else:
                # Add placeholder to maintain layout
                st.markdown("*No Y-axis needed for this chart type*")
                y_axis = None
        
        fig = create_plot(df, graph_type, x_axis, y_axis)
        st.plotly_chart(fig, use_container_width=True, key="preview_chart")
        
        if st.button("Save Visualization"):
            import time
            new_id = str(int(time.time() * 1000))
            
            new_graph = {
                'title': f"{graph_type} - {x_axis}" + (f" vs {y_axis}" if y_axis else ""),
                'type': graph_type,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'id': new_id,
                'config': {
                    'type': graph_type,
                    'x_axis': x_axis,
                    'y_axis': y_axis
                }
            }
            if 'saved_graphs' not in st.session_state:
                st.session_state.saved_graphs = []
            st.session_state.saved_graphs.append(new_graph)
            st.success("Visualization saved successfully!")
    
    with tab2:
        display_saved_graphs(df)

def display_saved_graphs(df):
    """Display saved graphs with filter support"""
    if not st.session_state.saved_graphs:
        st.info("No saved graphs yet. Create and save a graph to see it here!")
    else:
        # Use columns to display graphs in a grid
        cols = st.columns(2)
        
        # Store indices to remove after iteration
        to_remove = None
        
        for i, graph in enumerate(st.session_state.saved_graphs):
            # Ensure graph has an ID
            if 'id' not in graph:
                graph['id'] = str(i)
                
            with cols[i % 2]:
                with st.expander(f"{graph['title']}", expanded=True):
                    # Recreate the plot using the saved configuration and current filtered data
                    if 'config' in graph:
                        fig = create_plot(
                            df,
                            graph['config']['type'],
                            graph['config']['x_axis'],
                            graph['config'].get('y_axis')
                        )
                    else:
                        # For backward compatibility with older saved graphs
                        fig = create_plot(
                            df,
                            graph['type'],
                            graph['x_axis'],
                            graph.get('y_axis')
                        )
                    
                    unique_key = f"saved_graph_{graph['id']}"
                    st.plotly_chart(fig, use_container_width=True, key=unique_key)
                    
                    if st.button(f"Remove Graph", key=f"remove_{unique_key}"):
                        to_remove = i
        
        # Remove the graph outside the loop if needed
        if to_remove is not None:
            st.session_state.saved_graphs.pop(to_remove)
            st.rerun()

def create_plot(df, graph_type, x_axis, y_axis=None):
    """Create and return a Plotly figure"""
    if graph_type == "Bar Chart":
        if y_axis:
            fig = px.bar(df, x=x_axis, y=y_axis, 
                        title=f"Bar Chart of {y_axis} by {x_axis}")
        else:
            # Create count-based bar chart if no y_axis specified
            count_df = df[x_axis].value_counts().reset_index()
            fig = px.bar(count_df, x='index', y=x_axis, 
                        title=f"Count of {x_axis}")
            
    elif graph_type == "Pie Chart":
        counts = df[x_axis].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, 
                    title=f"Distribution of {x_axis}")
        
    elif graph_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, 
                     title=f"Line Chart of {y_axis} over {x_axis}")
        
    elif graph_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, 
                        title=f"Scatter Plot of {y_axis} vs {x_axis}")
        
    else:  # Histogram
        fig = px.histogram(df, x=x_axis,
                          title=f"Distribution of {x_axis}")
    
    return fig

def optimize_dataframe(df):
    """
    Optimize the dataframe by separating data into logical groups with proper deduplication
    """
    # Ensure Employee Code is treated as a unique identifier and strip spaces
    df['Employee Code'] = df['Employee Code'].astype(str).str.strip()
    
    # Helper function to clean and deduplicate a subset of data
    def clean_subset(data, columns, unique_keys):
        # Select only the specified columns
        subset = data[columns].copy()
        # Drop rows where all specified keys are null
        subset = subset.dropna(subset=unique_keys, how='all')
        # Drop duplicates based on specified keys, keeping the last occurrence
        subset = subset.drop_duplicates(subset=unique_keys, keep='last')
        return subset
    
    # Define column groups with their unique identifiers
    column_groups = {
        'core': {
            'columns': [
                'Employee Code', 'Employee Name', 'Birth Date', 
                'Gender', 'Ethnicity', 'Marital status', 'Citizenship',
                'Age', 'Age Range'
            ],
            'unique_keys': ['Employee Code']
        },
        'education': {
            'columns': [
                'Employee Code', 'Class 10 Percentage', 'Class 12 Percentage',
                'Highest Education Level', 'College / Institution - UG',
                'UG College affiliated to', 'UG Degree', 'UG CGPA', 'UG YOP',
                'PG/Diploma Course Institution/University', 'PG/Diploma Course Specialization',
                'PG/Diploma Course CGPA', 'PG YOP'
            ],
            'unique_keys': ['Employee Code']
        },
        'employment': {
            'columns': [
                'Employee Code', 'Legal Entity', 'Branch / Location', 'Business Unit',
                'Department', 'Designation', 'Employment type', 'Grade',
                'Reporting Manager', 'Facility - Seat Number / Remote',
                'Joining Date', 'Last Working Date', 'Resignation Date',
                'Status of employment(Active / Inactive / Suspended)',
                'Sub-Status of employment( Loss-of-Pay / Absconded / Resigned / Sabbatical)',
                'Overall Work Experience(in Years)', 'Tenure(in years)', 'Tenure Range'
            ],
            'unique_keys': ['Employee Code']
        },
        'payroll': {
            'columns': [
                'Employee Code', 'Payroll Month', 'Payroll currency',
                'Monthly Basic', 'Monthly House Rent Allowance', 'Dearness Allowance',
                'Monthly Allowance', 'PF - Employer', 'PF - Employee', 'ESI',
                'Monthly Special bonus', 'Monthly Hardship Allowance',
                'Annual Bonus Amount - Monthly Accrual', 'Gratuity - Monthly Accrual',
                'Monthly Deductions', 'Monthly Net Compensation', 'Monthly Gross Compensation',
                'Monthly CTC', 'Overtime hours', 'Overtime pay', 'Overtime payment'
            ],
            'unique_keys': ['Employee Code', 'Payroll Month']
        },
        'performance': {
            'columns': [
                'Employee Code', 'Payroll Month',
                '2022 Performance ratings', '2022 Feedback from performance reviews',
                '2023 Performance ratings', '2023 Feedback from performance reviews',
                'Key performance indicators (KPIs)', 'Attendance records',
                'Absence frequency and duration', 'Timeoff requests and approvals',
                'Availed Casual Leave', 'Availed Sick Leave', 'Availed Vacation Leave',
                'Availed Loss of Pay Days'
            ],
            'unique_keys': ['Employee Code', 'Payroll Month']
        }
    }
    
    # Process each group and store in a dictionary
    optimized_dfs = {}
    for group_name, group_config in column_groups.items():
        # Get columns that exist in the dataframe
        valid_columns = [col for col in group_config['columns'] if col in df.columns]
        valid_keys = [key for key in group_config['unique_keys'] if key in df.columns]
        
        if valid_columns and valid_keys:
            optimized_dfs[group_name] = clean_subset(
                df, 
                valid_columns, 
                valid_keys
            )
    
    return optimized_dfs

def csv_to_sqlite(df, table_name='data'):
    """Convert pandas DataFrame to SQLite database with cleaned column names"""
    conn = sqlite3.connect(':memory:')
    
    # Create optimized dataframes with cleaned column names
    tables = optimize_dataframe(df)
    
    # Create tables in SQLite
    for table_name, table_df in tables.items():
        table_df.to_sql(table_name, conn, index=False)
    
    return conn

def get_table_schema(df):
    """Generate schema strings for all tables"""
    tables = optimize_dataframe(df)
    schemas = []
    
    for table_name, table_df in tables.items():
        cols = [f"{col} ({str(table_df[col].dtype)})" for col in table_df.columns]
        schemas.append(f"{table_name}: {', '.join(cols)}")
    
    return '\n'.join(schemas)

def get_feedback_mappings():
    """
    Define mappings between simple terms and actual feedback phrases.
    Add more mappings based on your actual feedback data.
    """
    return {
        'Unacceptable': [
            'Unacceptable: Performance is consistently below expectations. Immediate improvement is needed. Consider additional training or reassignment.'
        ],
        'very poor': [
            'Very Poor: Performance is significantly below expectations. There are major gaps that need addressing. Regular follow-up and support required.'
        ],
        'Poor': [
            'Poor: Performance does not meet expectations. Improvement is needed in several areas. Specific action plan required.'
        ],
        'below average': [
            'Below Average: Performance is below expectations in some areas. Shows potential for improvement with guidance and support.'
        ],
        'average': [
            'Average: Meets basic job requirements but lacks consistency. Some areas need improvement to meet full expectations.'
        ],
        'Satisfactory': [
            'Satisfactory: Generally meets expectations but has room for growth. Regular performance reviews and development plans recommended.'
        ],
        'good': [
            'Good: Consistently meets expectations and occasionally exceeds them. Demonstrates solid performance with room for minor improvements.'
        ],
        'very good': [
            'Very Good: Often exceeds expectations. Shows strong performance with potential for further growth and development.'
        ],
        'excellent': [
            'Excellent: Consistently exceeds expectations. Demonstrates high-level performance and contributes significantly to team and organizational goals.'
        ],
        'Outstanding': [
            'Outstanding: Far exceeds expectations in all areas. Performance is exceptional and serves as a model for others. Consider for leadership roles and additional responsibilities.'
        ],
    }

def enhance_sql_query_with_mapping(natural_query, sql_query):
    """
    Enhance SQL query by replacing simple terms with mapped feedback phrases
    """
    mappings = get_feedback_mappings()
    
    for term, phrases in mappings.items():
        if term.lower() in natural_query.lower():
            conditions = []
            for phrase in phrases:
                conditions.append(f'''perf." 2022 Feedback from performance reviews" = '{phrase}' 
                                   OR perf." 2023 Feedback from performance reviews" = '{phrase}' ''')
            
            # Replace simple WHERE condition with mapped conditions
            where_start = sql_query.lower().find('where')
            if where_start != -1:
                base_query = sql_query[:where_start + 5]
                new_conditions = ' OR '.join(conditions)
                # Remove any backticks or 'sql' prefix that might be present
                clean_base_query = base_query.replace('`sql', '').replace('`', '').strip()
                sql_query = f"{clean_base_query} ({new_conditions})"
            else:
                # If there's no WHERE clause, add one
                new_conditions = ' OR '.join(conditions)
                # Remove any backticks or 'sql' prefix that might be present
                clean_query = sql_query.replace('`sql', '').replace('`', '').strip()
                sql_query = f"{clean_query} WHERE ({new_conditions})"
    
    return sql_query

def convert_to_sql(llm, natural_query, table_schemas):
    """Convert natural language query to SQL with comprehensive schema awareness"""
    try:
        system_template = """You are an expert SQL generator. You strictly return only SQL queries, nothing else.
        When generating SQL queries, follow these rules:
        1. Always include Employee Code in the SELECT clause
        2. Always include Payroll Month in the SELECT clause if the query involves payroll or salary data
        3. Always include ALL necessary JOIN clauses for any tables referenced in the SELECT clause
        4. Always enclose column names containing spaces in double quotes
        5. Use appropriate table aliases consistently: 
           - core as c
           - education as e
           - employment as emp
           - payroll as p
           - performance as perf
        6. Always use LEFT JOIN to preserve data when joining tables
        7. Join conditions should always use "Employee Code"
        8. Include exact column names as specified in the schema
        9. Qualify all column references with table alias (e.g., c."Employee Code")
        10. Do not use * in SELECT clause - explicitly list all columns
        11. For salary or payroll related queries, always include p."Payroll Month" in the SELECT clause"""

        human_template = """Convert this natural language query to SQL, ensuring ALL necessary table joins are included:
        Schema details:
        {schemas}

        Query to convert:
        "{natural_query}"
        
        Remember to:
        1. Always include c."Employee Code"
        2. Include p."Payroll Month" if query involves payroll/salary
        3. Use exact column names with double quotes
        4. Qualify all columns with table aliases"""

        chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]) | llm | StrOutputParser()

        sql_query = chain.invoke({
            "schemas": table_schemas,
            "natural_query": natural_query
        }).strip()

        return validate_and_fix_joins(sql_query)
    except Exception as e:
        return f"Error: {str(e)}"

def execute_fixed_sql_query(conn, query):
    """Execute SQL query with improved handling of duplicates and payroll data"""
    try:
        # Check if query involves payroll/performance data
        involves_monthly_data = ('payroll' in query.lower() or 'monthly' in query.lower() or 
                               'salary' in query.lower() or 'performance' in query.lower())
        
        # Fix query to handle time-series data properly
        fixed_query = fix_monthly_data_joins(query) if involves_monthly_data else query
        
        # Execute the fixed query
        df = pd.read_sql_query(fixed_query, conn)
        
        # Handle duplicates based on query type
        if involves_monthly_data:
            # Keep Payroll Month for salary/performance related queries
            group_cols = ['Employee Code']
            if 'Payroll Month' in df.columns:
                group_cols.append('Payroll Month')
        else:
            # Remove Payroll Month for non-salary queries to avoid duplicates
            group_cols = ['Employee Code']
            if 'Payroll Month' in df.columns:
                df = df.drop(columns=['Payroll Month'])
        
        # Group by appropriate columns and aggregate
        non_group_cols = [col for col in df.columns if col not in group_cols]
        if non_group_cols:
            # Define aggregation methods based on column type
            agg_dict = {}
            for col in non_group_cols:
                if df[col].dtype in ['int64', 'float64']:
                    agg_dict[col] = 'mean'  # Use mean for numeric columns
                else:
                    agg_dict[col] = 'first'  # Use first for non-numeric columns
            
            df = df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        return df
        
    except sqlite3.OperationalError as e:
        st.error(f"SQL Error: {str(e)}")
        print(f"Query that caused error: {fixed_query}")  # Debug print
        return None
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        print(f"Query that caused error: {fixed_query}")  # Debug print
        return None
        
    except sqlite3.OperationalError as e:
        st.error(f"SQL Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def fix_monthly_data_joins(query):
    """
    Fix JOIN conditions for monthly data tables (payroll and performance)
    """
    # Check if query involves monthly data tables
    if 'payroll AS p' in query and 'performance AS perf' in query:
        # Split query into parts
        select_part = query[:query.lower().find(' from ')]
        from_part = query[query.lower().find(' from '):]
        
        # Modify JOIN conditions to match on both Employee Code and Payroll Month
        from_part = from_part.replace(
            'LEFT JOIN payroll AS p ON c."Employee Code" = p."Employee Code"',
            'LEFT JOIN payroll AS p ON c."Employee Code" = p."Employee Code"'
        )
        from_part = from_part.replace(
            'LEFT JOIN performance AS perf ON c."Employee Code" = perf."Employee Code"',
            'LEFT JOIN performance AS perf ON c."Employee Code" = perf."Employee Code" AND p."Payroll Month" = perf."Payroll Month"'
        )
        
        # Reconstruct query
        fixed_query = select_part + from_part
        return fixed_query
    
    return query

def clean_query_results(df):
    """
    Clean query results by removing duplicates and handling monthly data properly
    """
    # Remove duplicate columns (keeping first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # If we have monthly data, ensure proper grouping
    if 'Payroll Month' in df.columns:
        # Group by Employee Code and Payroll Month to avoid duplicates
        group_cols = ['Employee Code', 'Payroll Month']
        non_group_cols = [col for col in df.columns if col not in group_cols]
        
        # For non-monthly columns, take the first value
        agg_dict = {col: 'first' for col in non_group_cols}
        
        # Group and aggregate
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    else:
        # For non-monthly queries, group by Employee Code only
        df = df.groupby('Employee Code', as_index=False).first()
    
    return df

def normalize_column_names(query):
    """Normalize column names in SQL query"""
    # Dictionary of correct column names
    column_mappings = {
        # Performance review column names
        '"2022 Performance ratings"': '"2022 Performance ratings"',
        '"2022 Feedback from performance reviews"': '"2022 Feedback from performance reviews"',
        '"2023 Performance ratings"': '"2023 Performance ratings"',
        '"2023 Feedback from performance reviews"': '"2023 Feedback from performance reviews"',
        '"Key performance indicators (KPIs)"': '"Key performance indicators (KPIs)"',
        '"Attendance records"': '"Attendance records"',
        '"Absence frequency and duration"': '"Absence frequency and duration"',
        '"Timeoff requests and approvals"': '"Timeoff requests and approvals"',
        
        # Overtime column names
        '"Overtime hours"': '"Overtime hours"',
        '"Overtime pay"': '"Overtime pay"',
        '"Overtime payment"': '"Overtime payment"'
    }
    
    normalized_query = query
    
    # Replace column names
    for incorrect, correct in column_mappings.items():
        normalized_query = normalized_query.replace(incorrect, correct)
    
    return normalized_query

def fix_query_column_names(query):
    """
    Fix common column name issues in SQL queries
    """
    # Clean the query
    clean_query = query.strip()
    if clean_query.startswith('`sql'):
        clean_query = clean_query[4:]
    clean_query = clean_query.replace('`', '')
    
    # Fix spaces in column references
    clean_query = re.sub(r'([a-z]+)\.\s*"([^"]+)"', r'\1."\2"', clean_query)
    
    # Fix spacing around SQL keywords
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'GROUP BY', 'ORDER BY']
    for keyword in sql_keywords:
        pattern = r'\s*\b' + keyword + r'\b\s*'
        clean_query = re.sub(pattern, f' {keyword} ', clean_query)
    
    # Clean up multiple spaces
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    
    return clean_query

def validate_and_fix_joins(sql_query):
    """Validate and fix JOIN clauses in the SQL query"""
    try:
        # Clean up the input query
        sql_query = sql_query.replace('`sql', '').replace('`', '').strip()
        
        # Fix column names
        sql_query = normalize_column_names(sql_query)
        
        # Remove duplicate quotes
        sql_query = sql_query.replace('""', '"')
        
        # Extract query parts
        parts = sql_query.split(' FROM ', 1)
        if len(parts) != 2:
            return sql_query
            
        select_part, rest = parts
        
        # Clean up JOIN conditions
        join_pattern = r'JOIN\s+(\w+)\s+(\w+)\s+ON'
        rest = re.sub(join_pattern, lambda m: f'JOIN {m.group(1)} {m.group(2)} ON', rest)
        
        # Table aliases
        table_aliases = {
            'core': 'c',
            'education': 'e',
            'employment': 'emp',
            'payroll': 'p',
            'performance': 'perf'
        }
        
        # Build query with proper JOIN structure
        joins = [f'FROM core {table_aliases["core"]}']
        for table, alias in table_aliases.items():
            if table != 'core' and alias in select_part:
                joins.append(
                    f'LEFT JOIN {table} {alias} ON '
                    f'{table_aliases["core"]}."Employee Code" = {alias}."Employee Code"'
                )
        
        # Add WHERE clause if present
        where_part = ''
        if 'WHERE' in rest:
            where_part = 'WHERE ' + rest.split('WHERE', 1)[1]
        
        # Combine all parts
        fixed_query = f'{select_part} {" ".join(joins)} {where_part}'.strip()
        
        return fixed_query

    except Exception as e:
        st.error(f"Error in validate_and_fix_joins: {str(e)}")
        return sql_query

def generate_insights(llm, query_results, natural_query, sql_query):
    """Generate detailed natural language insights from query results with TPM management"""
    try:
        data_summary = query_results.to_string()
        
        # Keep the original detailed prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR data analyst specializing in workforce analytics and employee data interpretation. 
            Your role is to provide clear, actionable insights that help business leaders make informed decisions.
            
            Guidelines for analysis:
            1. Focus on patterns and trends that have business impact
            2. Highlight any unusual or noteworthy findings
            3. Compare values against typical industry standards when possible
            4. Consider multiple dimensions of analysis (demographic, performance, compensation, etc.)
            5. Provide context for numerical findings
            6. Suggest potential actions or recommendations when appropriate"""),
            
            ("human", """Analyze the following HR data query results and provide comprehensive insights.

            Original Question: {question}
            SQL Query Used: {sql}
            
            Data Summary:
            {data}
            
            Please provide a structured analysis including:
            
            1. SUMMARY OF FINDINGS:
            - Key metrics and their significance
            - Overall patterns observed
            - Data coverage and completeness
            
            2. DETAILED ANALYSIS:
            - Breakdown of main metrics and their relationships
            - Statistical observations (averages, ranges, distributions)
            - Comparative analysis (across departments, roles, or time periods if applicable)
            - Identification of outliers or unusual patterns
            
            3. BUSINESS IMPLICATIONS:
            - Impact on workforce management
            - Potential areas of concern or opportunity
            - Cost implications (if relevant)
            - Compliance considerations (if applicable)
            
            4. ACTIONABLE RECOMMENDATIONS:
            - Specific steps that could be taken based on findings
            - Areas requiring further investigation
            - Potential interventions or policy adjustments
            
            5. DATA CONTEXT:
            - Confidence level in the findings
            - Suggestions for additional data points that could enhance the analysis""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            # Try without chunking first
            insights = chain.invoke({
                "question": natural_query,
                "sql": sql_query,
                "data": data_summary
            })
            return insights
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "token limit" in str(e).lower():
                return analyze_with_tpm_management(data_summary, natural_query, sql_query, llm)
            raise e

    except Exception as e:
        return f"Error generating insights: {str(e)}"

def analyze_with_tpm_management(data_summary, natural_query, sql_query, llm, chunk_size=80000):
    """Analyze data with TPM management by processing in chunks with delays"""
    import time
    
    # Show progress indicator
    progress_text = "Analyzing data in chunks..."
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Split data into chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in data_summary.split('\n'):
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Process each chunk with TPM management
    chunk_analyses = []
    
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR data analyst. Analyze this portion of data thoroughly but concisely.
        Focus on key patterns and metrics, remembering this is only part of the complete dataset."""),
        ("human", """Analyze this data segment:
        
        Question Context: {question}
        Data Segment ({chunk_num} of {total_chunks}):
        {chunk}
        
        Provide key findings focusing on:
        1. Key metrics and patterns
        2. Notable trends
        3. Important observations""")
    ])
    
    # Process chunks with progress updates
    for i, chunk in enumerate(chunks, 1):
        try:
            status_text.text(f"Processing chunk {i} of {len(chunks)}...")
            progress_bar.progress(i / len(chunks))
            
            # Add delay between chunks to manage TPM
            if i > 1:
                time.sleep(3)  # 3-second delay between chunks
                
            chain = chunk_prompt | llm | StrOutputParser()
            chunk_analysis = chain.invoke({
                "question": natural_query,
                "chunk": chunk,
                "chunk_num": i,
                "total_chunks": len(chunks)
            })
            
            # Store chunk analysis with metadata
            chunk_analyses.append({
                'chunk_num': i,
                'analysis': chunk_analysis
            })
            
        except Exception as e:
            st.warning(f"Warning: Error processing chunk {i}: {str(e)}")
            time.sleep(5)  # Longer delay after error
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Combine and synthesize all chunk analyses
    if chunk_analyses:
        st.info("Synthesizing results from all chunks...")
        
        # Sort analyses by chunk number to maintain order
        sorted_analyses = sorted(chunk_analyses, key=lambda x: x['chunk_num'])
        
        # Create combined analyses string
        combined_analyses = "\n\n".join([
            f"=== Chunk {a['chunk_num']} Analysis ===\n{a['analysis']}"
            for a in sorted_analyses
        ])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing multiple HR data analyses into a clear, actionable report.
            Focus on creating a coherent narrative from the separate analyses while maintaining analytical rigor."""),
            ("human", """Synthesize these separate analyses into one comprehensive report:
            
            Original Question: {question}
            SQL Query: {sql}
            
            Individual Analyses:
            {analyses}
            
            Provide a structured analysis including:
            1. Key Findings Summary
            2. Detailed Analysis
            3. Business Implications
            4. Recommendations
            5. Data Quality Notes""")
        ])
        
        try:
            # Add delay before final synthesis
            time.sleep(3)
            
            synthesis_chain = synthesis_prompt | llm | StrOutputParser()
            final_analysis = synthesis_chain.invoke({
                "question": natural_query,
                "sql": sql_query,
                "analyses": combined_analyses
            })
            
            st.success("Analysis complete!")
            return final_analysis
            
        except Exception as e:
            st.error(f"Error in final synthesis: {str(e)}")
            # If synthesis fails, display individual chunk analyses
            st.warning("Showing individual chunk analyses due to synthesis error:")
            return combined_analyses
    
    return "Unable to generate insights. No valid analyses were produced from the data chunks."
    
def create_query_visualization(df, graph_type, x_axis, y_axis=None):
    """Create visualization for query results"""
    if graph_type == "Bar Chart":
        if y_axis:
            fig = px.bar(df, x=x_axis, y=y_axis, 
                        title=f"Bar Chart of {y_axis} by {x_axis}")
        else:
            count_df = df[x_axis].value_counts().reset_index()
            fig = px.bar(count_df, x='index', y=x_axis, 
                        title=f"Count of {x_axis}")
            
    elif graph_type == "Pie Chart":
        counts = df[x_axis].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, 
                    title=f"Distribution of {x_axis}")
        
    elif graph_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, 
                     title=f"Line Chart of {y_axis} over {x_axis}")
        
    elif graph_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, 
                        title=f"Scatter Plot of {y_axis} vs {x_axis}")
        
    else:  # Histogram
        fig = px.histogram(df, x=x_axis,
                          title=f"Distribution of {x_axis}")
    
    return fig

def init_query_session_state(result_df):
    """Initialize session state for query visualization"""
    if 'query_result_df' not in st.session_state:
        st.session_state['query_result_df'] = None
    if 'query_graph_type' not in st.session_state:
        st.session_state['query_graph_type'] = "Bar Chart"
    if 'query_x_axis' not in st.session_state:
        st.session_state['query_x_axis'] = None
    if 'query_y_axis' not in st.session_state:
        st.session_state['query_y_axis'] = None
    if 'show_viz' not in st.session_state:
        st.session_state['show_viz'] = False
    if 'current_query' not in st.session_state:
        st.session_state['current_query'] = None
        
    # Only update if new results are provided
    if result_df is not None:
        st.session_state['query_result_df'] = result_df
        if len(result_df.columns) > 0:
            st.session_state['query_x_axis'] = result_df.columns[0]
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            st.session_state['query_y_axis'] = numeric_cols[0] if len(numeric_cols) > 0 else None
        st.session_state['show_viz'] = True

def show_visualization_options(result_df):
    """Display visualization options for query results"""
    if result_df is None:
        return
        
    st.subheader("Customize Visualization")
    
    # Generate unique keys for each widget
    prefix = "query_viz_"
    
    # Select visualization type
    new_graph_type = st.selectbox(
        "Select Graph Type",
        ["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot", "Histogram"],
        index=["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot", "Histogram"].index(st.session_state['query_graph_type']),
        key=f"{prefix}graph_type"
    )
    
    # Create columns for axis selection
    col1, col2 = st.columns(2)
    with col1:
        new_x_axis = st.selectbox(
            "Select X-axis", 
            result_df.columns,
            index=list(result_df.columns).index(st.session_state['query_x_axis']) if st.session_state['query_x_axis'] in result_df.columns else 0,
            key=f"{prefix}x_axis"
        )
    
    with col2:
        if new_graph_type not in ["Pie Chart", "Histogram"]:
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                default_y_index = (list(numeric_cols).index(st.session_state['query_y_axis']) 
                                 if st.session_state['query_y_axis'] in numeric_cols 
                                 else 0)
                new_y_axis = st.selectbox(
                    "Select Y-axis",
                    numeric_cols,
                    index=default_y_index,
                    key=f"{prefix}y_axis"
                )
            else:
                st.warning("No numeric columns available for Y-axis")
                new_y_axis = None
        else:
            st.markdown("*No Y-axis needed for this chart type*")
            new_y_axis = None
    
    # Update session state
    st.session_state['query_graph_type'] = new_graph_type
    st.session_state['query_x_axis'] = new_x_axis
    st.session_state['query_y_axis'] = new_y_axis
    
    # Create and display the visualization
    try:
        fig = create_query_visualization(result_df, new_graph_type, new_x_axis, new_y_axis)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def main():
    """Main function to run the integrated tool"""
    llm = get_llm_client()
    
    # Initialize session state
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'saved_graphs' not in st.session_state:
        st.session_state.saved_graphs = []
    
    df = load_data()
    
    if df is not None:
        # Create filters in sidebar
        filters = create_filters(df)
        filtered_df = apply_filters(df, filters) if filters else df
        
        st.write("### Dashboard")
        st.dataframe(filtered_df)
        
        # Create tabs with Visualize as default
        tab2, tab1 = st.tabs(["Visualize", "Query"])
        
        with tab1:
            natural_language_query_section(filtered_df, llm)
        
        with tab2:
            visualization_section(filtered_df)
    else:
        st.warning("Please upload a CSV file to continue.")

if __name__ == "__main__":
    main()
