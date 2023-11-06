import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import requests
import pickle
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import itertools
import re
import streamlit.components.v1 as components
from pyvis.network import Network
from bs4 import BeautifulSoup
from wordcloud import WordCloud

# Setting Random Seed
random.seed(42)
# Function to load a pickle file and extract titles
def load_pickle(file_path):
    """
    Load data from a pickle file.

    :param file_path: The path to the output pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    return data

# Full Profile Information on DR-NTU
def prof_profile(prof):
    global df
    df_prof = df.loc[df['Full Name'] == prof, :]
    for i, row in df_prof.iterrows():
        url = row['DR-NTU URL']
        soup_source = requests.get(url).text
        soup = BeautifulSoup(soup_source,'html.parser')
        
        if (soup.find('div', id='biographyDiv')) == None:
            background = "Not Applicable"
        else:
            background = soup.find('div', id='biographyDiv').text.strip()
        
        if (soup.find('div', id='researchinterestsDiv')) == None:
            research_interest = "Not Applicable"
        else:
            research_interest = soup.find('div', id='researchinterestsDiv').text.strip()
            
        no_citations = row['Citations Count']
        
        return i, background, research_interest, no_citations
    
# Get Ranks of Conferences of Publications
def get_rank(venue):
    if '(' in venue and venue.split('(')[1].split(')')[0] != '':
        venue_part = venue.split('(')[1].split(')')[0]
    else:
        venue_part = venue

    url = f'http://portal.core.edu.au/conf-ranks/?search={venue_part.replace(" ", "%20")}&by=all&source=all&sort=atitle&page=1'
    soup_source = requests.get(url).text
    soup = BeautifulSoup(soup_source, 'html.parser')
    
    def refine_string(s):
        s = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", "", s)
        return s
    
    if len(soup.find_all('table')) == 1:
        for row in soup.find_all('tr')[1:]:
            title = row.find_all('td')[0].text.strip()
            acronym = row.find_all('td')[1].text.strip()
            rank = row.find_all('td')[3].text.strip()
            if venue == title or venue.upper() == acronym \
                or acronym.upper() in [refine_string(v).upper() for v in venue.split()] \
                or venue.upper() in [refine_string(t).upper() for t in title.split()]:
                return rank
    return 'unranked'

# Function to display the home details
def display_home(prof, background):
    # Display profile
    st.image('ntulogo.png', width=650)
    profile_col , profile_col2 = st.columns(2)
    with profile_col:
        st.image(f'profile_pics/{prof}.jpg', width = 250)
    butcol1, butcol2, butcol3 = st.columns(3)
    with butcol1:
        if 'DR-NTU URL' in df.columns:
                dr_ntu_url = df.loc[df['Full Name'] == prof, 'DR-NTU URL'].values[0]
                if dr_ntu_url:
                    if st.button(f"DR-NTU Link"):
                        webbrowser.open_new_tab(dr_ntu_url)
    with butcol2:
        if 'Google Scholar URL' in df.columns:
            google_scholar_url = df.loc[df['Full Name'] == prof, 'Google Scholar URL'].values[0]
            if google_scholar_url:
                if st.button(f"Google Scholar Link"):
                    webbrowser.open_new_tab(google_scholar_url)
    with butcol3:
        if 'Website URL' in df.columns:
            website_url = df.loc[df['Full Name'] == prof, 'Website URL'].values[0]
            if website_url:
                if st.button(f"Personal Website Link"):
                    webbrowser.open_new_tab(website_url)
    profile_container = st.container()
    profile_container.title(prof)
    profile_container.subheader('Biography')
    profile_container.write(background)
    col1, col2 = profile_container.columns([3,2])
    # Second Column
    with col1:
        st.subheader('Education')
        education = load_pickle(f'education_set/education_{prof}.pkl')
        for key in education:
            level_of_education = key['Level of Education']
            course = key['Name of Course']
            attained_at = key['Attained at']
            st.write(f'Level of Education: {level_of_education}')
            st.write(f'Name of Course: {course}')
            st.write(f'Attained at: {attained_at}')
            st.write("\n")
    with col2:
        st.subheader('Research Interest')
        research = load_pickle(f'research_interest_set/interest_{prof}.pkl')
        st.write(research)
        st.subheader("Citations (Cited By)")
        st.write(no_citations)
    

def display_publication_details(publication):
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(publication)
    # Join authors into a single string
    st.dataframe(df, width=1000)

def display_publications(prof):
    # Display profile
    profile_container = st.container()
    profile_container.title('Publications')
    publications = load_pickle(f'publication_set/publications_{prof}.pkl')
    # Create a select box to choose between "View All" and "View by Topic"
    selection = st.selectbox("Select an option:", ["View All", "View by Topic", "View by Year"])
    #st.selectbox('Choose your publication to display', publications)
    if selection == "View All":
        display_publication_details(publications)
    elif selection == "View by Topic":
            topics = list(set([publication["subcategory"] for publication in publications]))
            selected_topic = st.multiselect("Select a topic to filter:", topics)
            
            selected_publications = [publication for publication in publications if publication["subcategory"] in selected_topic]
            display_publication_details(selected_publications)
    elif selection == "View by Year":
        years = sorted(set([publication["year"] for publication in publications]), reverse=True)
        selected_year = st.multiselect("Select year to filter:", years)
        
        selected_publications = [publication for publication in publications if publication["year"] in selected_year]
        display_publication_details(selected_publications)
# Show Citations Throughout the Year
def citations_plot(citations_year):
    labels = list(citations_year.keys())
    values = list(citations_year.values())
    labels, values = zip(*sorted(zip(labels, values)))

    colors = ['lightblue' if i != max(values) else 'red' for i in values]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors, width=0.6)

    for label, value in zip(labels, values):
        plt.text(label, value, str(value), ha='center', va='bottom')
    
    plt.xlabel('Years')
    plt.ylabel('Citations Count')
    plt.title('Citations Count per Year')
    plt.xticks(labels, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Stacked Bar Chart to show the count of publications with each category throughout the years
def bar_shift_interest(data):
    years = list(set(year for year, _ in data.keys()))
    categories = list(set(category for _, category in data.keys()))
    # Prepare the data for the stacked bar chart
    stacked_data = {category: [data.get((year, category), 0) for year in years] for category in categories}
    # Create a DataFrame from the data
    df = pd.DataFrame(stacked_data, index=years)
    # Create a stacked bar chart using Plotly Express
    fig = px.bar(
        df,
        x=df.index,
        y=categories,
        title='Stacked Bar Chart of Publications by Year and Category',
        category_orders={'Category': categories},
    )
    # Create sliders for height and width
    height = st.slider('Height', min_value=300, max_value=1000, value=500, step=50)
    width = st.slider('Width', min_value=300, max_value=1000, value=800, step=50)
    # Update the chart size based on user input
    fig.update_layout(height=height, width=width)
    # Customize the layout of the chart
    fig.update_layout(barmode='stack')
    # Change the legend title
    fig.update_layout(legend_title_text='Categories')
    fig.update_xaxes(title_text='Years')
    fig.update_yaxes(title_text='Count')
    return fig

# Count the occurances of each year type in the publications
def pubs_count_year(publications):
    year_counts = {}
    # Count the publications for each year
    for publication in publications:
        year = publication['year']
        if year in year_counts:
            year_counts[year] += 1
        else:
            year_counts[year] = 1
    return year_counts

# Count the occurances of each type in the publications
def pubs_count_type(publications):
    type_counts = {}
    for publication in publications:
        pub_type = publication['type']
        if pub_type in type_counts:
            type_counts[pub_type] += 1
        else:
            type_counts[pub_type] = 1
    return type_counts

# Count the occurances of each subtopic in the publications
def pubs_count_subtopic(publications):
    sub_counts = {}
    for publication in publications:
        sub_topic = publication['subcategory']
        if sub_topic in sub_counts:
            sub_counts[sub_topic] += 1
        else:
            sub_counts[sub_topic] = 1
    return sub_counts
# Count the Number of Ranks for publications
def pubs_count_rank(publications):
    rank_counts = {}
    ranking_order = ['A*', 'A', 'B', 'C', 'D', 'E']
    
    for publication in publications:
        rank = publication['venue_rank']
        
        if rank in ranking_order:
            if rank in rank_counts:
                rank_counts[rank] += 1
            else:
                rank_counts[rank] = 1
    
    return rank_counts
# Count total SCSE Pubs
def count_total_SCSE_pubs():
    yearly_counts = {}
    
    for _, row in df.iterrows():
        name = row['Full Name']
        publications = load_pickle(f'publication_set/publications_{name}.pkl')
        for publication in publications:
            year = publication["year"]
            if year in yearly_counts:
                yearly_counts[year] += 1
            else:
                yearly_counts[year] = 1
                
    # Sort the dictionary by year in ascending order.
    sorted_yearly_counts = {k: v for k, v in sorted(yearly_counts.items(), reverse = True)}

    return sorted_yearly_counts


# Count each combinations of year and subcategory for each publication
def pubs_count_interests(publications):
    interest_counts = {}
    # Count the publications for each year and subcategory
    for publication in publications:
        year = publication['year']
        subcategory = publication['subcategory']
        # Create a tuple of year and subcategory to use as a key
        key = (year, subcategory)
        
        if key in interest_counts:
            interest_counts[key] += 1
        else:
            interest_counts[key] = 1
    return interest_counts
    
# Menu for Publications Visualisations (Individual)
def display_publications_visualisations(prof):
    publications = load_pickle(f'publication_set/publications_{prof}.pkl')
    citations_data = load_pickle(f'citations_yearly_set/citations_{prof}.pkl')
    # Create a select box to choose 
    selection = st.selectbox("Select an option:",
                              ["View Citations Count Plot", "View Publication Count Plot", "View Type of Publications Count Plot",
                               "View Quality of Venues Plot"])
    if selection == "View Citations Count Plot":
        # Extract the years and counts from the dictionary
        labels = list(citations_data.keys())
        values = list(citations_data.values())
        # Sliders to adjust chart parameters
        bargap = st.slider('Bar Gap',min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height = st.slider('Chart Height',min_value=100, max_value=800, value=400, step=50)
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Publication Count Over the Years', labels={'x': 'Year', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of citations of all the publications by {prof} over the years")
    elif selection == "View Publication Count Plot":
        pub_dict = pubs_count_year(publications)
        # Extract the years and counts from the dictionary
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Sliders to adjust chart parameters
        bargap = st.slider('Bar Gap', min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height = st.slider('Chart Height', min_value=100, max_value=800, value=400, step=50)
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Publication Count Over the Years', labels={'x': 'Year', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of publications by {prof} over the years")
    elif selection == "View Type of Publications Count Plot":
        pub_dict = pubs_count_type(publications)
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Sliders to adjust chart parameters
        bargap = st.slider('Bar Gap', min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height = st.slider('Chart Height', min_value=100, max_value=800, value=400, step=50)
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Number of Publications per Type', labels={'x': 'Type of Publication', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of publications for each type by {prof} over the years")
    elif selection == "View Quality of Venues Plot":
        pub_dict = pubs_count_rank(publications)
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Sliders to adjust chart parameters
        bargap = st.slider('Bar Gap', min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height = st.slider('Chart Height', min_value=100, max_value=800, value=400, step=50)
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Quality of Venues (Publications)', labels={'x': 'Ranking', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the quality of venues for the publications by {prof} over the years")

def display_publications_visualisations_scse(prof, bargap, height, selection):
    publications = load_pickle(f'publication_set/publications_{prof}.pkl')
    citations_data = load_pickle(f'citations_yearly_set/citations_{prof}.pkl')
    if selection == "View Citations Count Plot":
        # Extract the years and counts from the dictionary
        labels = list(citations_data.keys())
        values = list(citations_data.values())
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Publication Count Over the Years', labels={'x': 'Year', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of citations of all the publications by {prof} over the years")
    elif selection == "View Publication Count Plot":
        pub_dict = pubs_count_year(publications)
        # Extract the years and counts from the dictionary
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Publication Count Over the Years', labels={'x': 'Year', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of publications by {prof} over the years")
    elif selection == "View Type of Publications Count Plot":
        pub_dict = pubs_count_type(publications)
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Number of Publications per Type', labels={'x': 'Type of Publication', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the number of publications for each type by {prof} over the years")
    elif selection == "View Quality of Venues Plot":
        pub_dict = pubs_count_rank(publications)
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Quality of Venues (Publications)', labels={'x': 'Ranking', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.write(f"The following graph displays the quality of venues for the publications by {prof} over the years")

# Function to match Full Name to DBLP Names
def match_fullname_to_dblp(df, full_name):
    match = df[df['Full Name'] == full_name]
    if not match.empty:
        dblp_name = match['DBLP Names'].values[0]
        return dblp_name
    else:
        return None

# Displaying the temporal characteristics of the professor
def display_analysis(prof):
    st.title("Analysis of " + str(prof))
    publications = load_pickle(f"publication_set/publications_{prof}.pkl")
    interest_counts = pubs_count_interests(publications)
    st.write(f"The following chart shows the Stacked Bar Chart of publications categorised by Subcategories and Year, this helps us visualise the shift in interest for {prof} over the years in research.")
    # Display the chart in Streamlit
    st.plotly_chart(bar_shift_interest(interest_counts))
    st.write(f"The following chart shows the Heatmap of publications categorised by Subcategories and Year, this helps us visualise the shift in interest for {prof} over the years in research.")
    data = interest_counts
    # Extract years and categories from the data
    years = list(set(year for year, _ in data.keys()))
    categories = list(set(category for _, category in data.keys()))

    # Prepare the data for the heatmap
    heatmap_data = {
        'Year': [year for year, _ in data.keys()],
        'Category': [category for _, category in data.keys()],
        'Count': [data.get((year, category), 0) for year, category in data.keys()]
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(heatmap_data)

    # Create a heatmap using Plotly Express
    fig = go.Figure(data=go.Heatmap(
        y=df['Category'],
        x=df['Year'],
        z=df['Count'],
        colorscale='Viridis',  # You can change the colorscale as needed
        colorbar_title='Count',
        hoverongaps=False,
    ))

    # Customize the layout of the chart
    fig.update_yaxes(title_text='Category')
    fig.update_xaxes(title_text='Year')
    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

# Function to find researchers with common research interests with a given researcher
def find_common_interests(selected_prof, specified_interest):
    common_interests = []
    # Find the research interests of the specified researcher
    researcher_interests = {}
    researcher_interests[selected_prof] = load_pickle(f"research_interest_set/interest_{selected_prof}.pkl")
    compare_interests = {}

    # Iterate through the DataFrame to find common interests
    for _, row in df.iterrows():
        compare_name = row['Full Name']
        compare_interests[compare_name] = load_pickle(f"research_interest_set/interest_{compare_name}.pkl")
        
    # Split the specified researcher's interests
    specified_interests_list = researcher_interests[selected_prof].split(', ')

    # Iterate through other researchers to find similar interests
    for compare_name, interests in compare_interests.items():
        if compare_name != selected_prof:
            # Split the interests of the current researcher
            compare_interests_list = interests.split(', ')

            # Check if the specified interest matches any of the interests of the current researcher (allowing partial matches)
            for interest in compare_interests_list:
                if specified_interest in interest:
                    common_interests.append(compare_name)
    return common_interests

def sum_interests():
    global df
    temp = {}
    for _, row in df.iterrows():
        name = row['Full Name']
        interests = load_pickle(f"research_interest_set/interest_{name}.pkl")
        temp[name] = interests
    return temp

# Displaying Collaboration network
def display_network(prof):
    st.title("Network of " + str(prof))
    dblp_name = match_fullname_to_dblp(df,prof)
    selection = st.sidebar.selectbox("Select an option:", ["View SCSE Network", "View Outside NTU Network", 
                                                           "Group By Research Interest"])
    if selection == "View SCSE Network":
        st.subheader("SCSE Collaborations")
        G = create_scse_graph(dblp_name, True)
        G.save_graph('scse_network_st.html')
        HtmlFile = open('scse_network_st.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=800, width = 1000)
        st.write(f"The following is the network graph of {prof} within SCSE.")
        st.write(f"Red Node denotes {prof}, Orange Node denotes the SCSE professors {prof} has collaborated with.")
    elif selection == "View Outside NTU Network":
        st.subheader("Outside NTU Collaborations")
        G = create_outsideNTU_graph(prof)
        G.save_graph('outsideNTU_network_st.html')
        HtmlFile = open('outsideNTU_network_st.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=800, width = 1000)
        st.write(f"The following is the network graph of {prof} outside of NTU.")
        st.write(f"Orange Node denotes the coauthors {prof} collaborated with")
    elif selection == "Group By Research Interest":
         # Find the research interests of the specified researcher
        interests = load_pickle(f"research_interest_set/interest_{prof}.pkl")
        # Split the string into a list of individual research interests
        interests_list = [interest.strip() for interest in interests.split(',')]
        selected_interest = st.sidebar.selectbox(f'Research Interests of {prof}:',interests_list)
        st.subheader(f"Researchers in SCSE interested in {selected_interest}")
        names_list = find_common_interests(prof, selected_interest)
        if len(names_list) > 10:
            # Create two columns to display names
            col1, col2 = st.columns(2)
            # Calculate the number of names in each column
            half_len = len(names_list) // 2
            # Display names in the first column
            with col1:
                for i in range(half_len):
                    st.write(names_list[i])
            # Display names in the second column
            with col2:
                for i in range(half_len, len(names_list)):
                    st.write(names_list[i])
        else:
            for i in names_list:
                st.write(i)

# This function is to create the graph for selected professor, if show all network exclude the color code part 
def create_scse_graph(selected_prof, color_code):
    global scse_authors
     # Define the set to store connected nodes
    connect_nodes = set()
    # Creating Pyvis Network
    G = Network(height='1000px', bgcolor='#FFFFFF', font_color='black')
     # Iterate through the authors and add them as nodes in the Pyvis graph (excluding "NaN")
    for prof_name in scse_authors:
        if not pd.isna(prof_name) and prof_name != 'NaN':
            # Add nodes with a larger size (adjust the value as needed)
            G.add_node(prof_name, label=prof_name)

    # Create edges between authors in the Pyvis graph
    for i, row in df.iterrows():
        name = row['Full Name']
        dblp_name = row['DBLP Names']
        if pd.isna(dblp_name) or dblp_name == 'NaN':
            continue  
        with open(f'publication_set/publications_{name}.pkl', 'rb') as f:
            publications = pickle.load(f)
        for pub in publications:
            authors = pub['authors']
            for author in authors:
                # This will add regular edges and exclude self-loops
                if author in scse_authors and author != dblp_name:
                    G.add_edge(dblp_name, author)  
    if color_code:         
        # Functions to Colour Code nodes based on selected Professors
        connect_nodes = set()
        for edge in G.edges:
            if edge['from'] == selected_prof:
                connect_nodes.add(edge['to'])
            elif edge['to'] == selected_prof:
                connect_nodes.add(edge['from'])

        for node in G.nodes:
            if node['label'] == selected_prof:
                node['color'] = '#FB0303'
            elif node['label'] in connect_nodes:
                node['color'] = '#FB8E03'
    else: 
        return G
    # Return the graph after creating and coloring nodes        
    return G
# Creating Network Graph Outside NTU
def create_outsideNTU_graph(selected_prof):
    publications = load_pickle(f'publication_set/publications_{selected_prof}.pkl')
    # Create a Pyvis Network instance
    net = Network(height='1000px', bgcolor='#242424', font_color='white')

    # Add nodes for Selected Professor and coauthors (excluding the excluded names)
    coauthors = set()

    for pub in publications:
        authors = pub['authors']
        for author in authors:
            if author != selected_prof and author not in scse_authors:
                coauthors.add(author)
    print(coauthors)
    # Add nodes for Selected Professor and coauthors (excluding the excluded names)
    net.add_node(selected_prof, color='orange', title=selected_prof)  
    for author in coauthors:
        net.add_node(author, title=author)  
        
    # Add edges to represent coauthorship
    for pub in publications:
        authors = pub['authors']
        for author in authors:
            if author != selected_prof and author not in scse_authors:
                net.add_edge(selected_prof, author)
    return net

# Main Page Content 
df = pd.read_csv('Lye_En_Lih_updated.csv')
sorted_df = df.sort_values('Full Name')
scse_authors = load_pickle('scse_network_authors.pkl')
scse_interests = sum_interests()
#st.markdown("<h1 style='text-align: left;'>NTU SCSE Faculty Member</h1>", unsafe_allow_html=True)
st.sidebar.image("ntu_sidelogo.png", width=350)
st.sidebar.header('NTU SCSE Faculty Member Dashboard')
menu_selection = st.sidebar.selectbox("Select an Option:",["Individual Faculty Profile", "SCSE Profile"])
if menu_selection == "Individual Faculty Profile":
    st.sidebar.subheader('Choose your Professor')
    prof = st.sidebar.selectbox('Research Profile:', sorted_df['Full Name'])
    prof_index, background, research_interest,  no_citations = prof_profile(prof)
    st.sidebar.subheader("Individual Faculty Navigation")
    selection = st.sidebar.radio("Go To", ["Profile Home", "Publications Visualisations", "Analysis of Professor", 
                                                "Collaboration Network"])
    # Side Bar Selectionsto display different pages
    if selection == "Profile Home":
        display_home(prof,background)
        display_publications(prof)
    elif selection == "Publications Visualisations":
        st.title("Publications Visualisations")
        display_publications_visualisations(prof)
    elif selection == "Analysis of Professor":
        display_analysis(prof)
    elif selection == "Collaboration Network":
        display_network(prof)
elif menu_selection == "SCSE Profile":
    st.sidebar.subheader("SCSE Faculty Navigation")
    scse_menu_selection = st.sidebar.radio("Go To", ["Profile Home", "Collaboration Network"])
    if scse_menu_selection == "Profile Home":
        st.header("SCSE Profile Page")
        pub_dict = count_total_SCSE_pubs()
        labels = list(pub_dict.keys())
        values = list(pub_dict.values())
        # Sliders to adjust chart parameters
        bargap = st.slider('Bar Gap', min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height= st.slider('Chart Height', min_value=100, max_value=800, value=400, step=50)
        # Create a bar chart using Plotly Express
        fig = px.bar(x=labels, y=values, title='Total Publications in SCSE', labels={'x': 'Years', 'y': 'Publications Count'})
        # Customize the layout of the chart
        fig.update_layout(
            bargap=bargap,  # Adjust the gap between bars
            height = height
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Word Cloud of Research Interests
        st.subheader("WordCloud of Research Interests of SCSE Members")
        df = pd.DataFrame(list(scse_interests.values()), columns=['Research Interests'])
        # Concatenate all research interests into a single string
        all_interests = " ".join(df['Research Interests'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_interests)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
        st.subheader("Compare Researchers")
        col1, col2 = st.columns(2)
        with col1:
            comp_prof = st.selectbox('Research Profile 1:', sorted_df['Full Name'])
            publications1 = load_pickle(f'publication_set/publications_{comp_prof}.pkl')
            subtopic = pubs_count_subtopic(publications1)
        with col2:
            comp_prof2 = st.selectbox('Research Profile 2:', sorted_df['Full Name'])
            publications2 = load_pickle(f'publication_set/publications_{comp_prof2}.pkl')
            subtopic2 = pubs_count_subtopic(publications2)
        bargap= st.slider('Bar Gap', key = 'comp1' ,min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        height= st.slider('Chart Height', key = 'comph1',min_value=100, max_value=800, value=400, step=50)
        selection = st.selectbox("Select an option:",
                                ["View Citations Count Plot", "View Publication Count Plot", "View Type of Publications Count Plot",
                                "View Quality of Venues Plot"])
        st.subheader(f"Researcher Profile {comp_prof}")
        display_publications_visualisations_scse(comp_prof, bargap, height, selection = selection)
        subtopic_data = [(category, value) for category, value in subtopic.items()]
        # Create a Plotly bar chart
        fig = px.bar(subtopic_data, x=1, y=0, labels={'0': 'Category', '1': 'Count'})
        st.plotly_chart(fig)
        st.subheader(f"Researcher Profile {comp_prof2}")
        display_publications_visualisations_scse(comp_prof2, bargap, height,selection = selection)
        # Convert the dictionary to a list of tuples for easier plotting
        subtopic_data_2 = [(category, value) for category, value in subtopic2.items()]
        # Create a Plotly bar chart
        fig = px.bar(subtopic_data_2, x=1, y=0, labels={'0': 'Category', '1': 'Count'})
        st.plotly_chart(fig)
        
        
    elif scse_menu_selection == "Collaboration Network":
        st.header("Compare Researchers Collaborations")
        col1, col2 = st.columns(2)
        with col1:
            comp_prof = st.selectbox('Research Profile 1:', sorted_df['Full Name'])
            dblp_name = match_fullname_to_dblp(df,comp_prof)
        G = create_scse_graph(dblp_name, True)
        G.save_graph('scse_network_comp.html')
        HtmlFile = open('scse_network_st.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=800, width = 1000)
        with col2:
            comp_prof2 = st.selectbox('Research Profile 2:', sorted_df['Full Name'])
            dblp_name2 = match_fullname_to_dblp(df,comp_prof2)
        G = create_scse_graph(dblp_name2, True)
        G.save_graph('scse_network_comp2.html')
        HtmlFile = open('scse_network_comp2.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=800, width = 1000)


