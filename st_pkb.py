import streamlit as st
from PIL import Image
from openai import OpenAI
import io
import pytesseract
import pandas as pd
import awswrangler as wr
import boto3
import base64
import json
import os
from urllib.parse import urlparse
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from taxon_parser import TaxonParser, UnparsableNameException
import math
import numpy as np

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "Not required in this demo."

st.title("🌍🤖️🏞🏷 PKB Herbarium Label Auto Digitisation")
st.caption("🚀 Herbarium Sheet Optical Character Recognition (OCR) Service Powered by AWS and OpenAI")

"""
This is a demo of PKB OCR
"""
"""
📷 1. Upload herbarium sheet - please keep the maximum width of the image to 1024 pixels.
"""
"""
🔍 2. Using GPT-4 to extract structured information from the herbarium sheet.
"""
"""
🌳 3. Aligning data to the PKB - interacting with our PKB Neptune Cloud service using BERT Encoder and Relational Graph Convolution Network (RGCN) for node classification and link prediction. This process will take a few minutes depending on the extracted data.
"""
"""
🗺️ 4. Graph Visualization — plotting a subgraph of the PKB related to the given specimen.
"""
"""
🗺️ 5. Graph Explorer — Navigator to the PKB. This PKB demo contains all Solanaceae preserved specimens from GBIF (cut-off date 01/04/2024). There are 3.3M nodes and 3.6M edges in total.
"""
st.link_button("Go to Graph Explorer", "https://18.134.179.85/explorer/#/graph-explorer")
"""
🗺️ 6. Return RDF data — for Ben to update.
"""

##############################################################################
##############################################################################
##############################################################################

url = "tf-20240207121511848700000014.cluster-c9ezntcdvm4p.eu-west-2.neptune.amazonaws.com"  # The Neptune Cluster endpoint
iam_enabled = True  # Set to True/False based on the configuration of your cluster
neptune_port = 8182  # Set to the Neptune Cluster Port, Default is 8182
region_name = "eu-west-2"  # Replace with your Neptune cluster's region
endpoint='collecto-2024-07-19-07-17-490000-endpoint'

# Create a session with the specified region
session = boto3.Session(region_name=region_name)

client = wr.neptune.connect(url, neptune_port, iam_enabled=iam_enabled, boto3_session=session)
st.text("Connecting to Neptune......")
st.text(client.status())

# Set your OpenAI API key
OpenAI.api_key = ""
gpt_client = OpenAI(api_key=OpenAI.api_key) #Best practice needs OPENAI_API_KEY environment variable

promt = """
Find the collector's name, taxon, country/location(convert into ISO Alpha-2 country, which only has 252 Countries), institution(convert into institution code), and year. 
Please return only content and its classes in the following order: collector name, taxon, country/location, ISO, institution, institution code, and year.
Show the result in the below format:
collector name: 
taxon: 
country/location: 
ISO: 
institution: 
institution code: 
year: 
"""

##############################################################################
##############################################################################
##############################################################################

def convert_to_dataframe(input_string):
    # Split the input string into lines
    lines = input_string.strip().split('\n')   
    # Initialize an empty dictionary to store the key-value pairs
    data = {}
    
    # Process each line
    for line in lines:
        key, value = line.split(': ', 1)
        # Convert the value to string and strip it
        data[key.strip().replace(' ', '_').lower()] = str(value.strip())
    
    # Map the keys to the expected row names
    mapping = {
        'collector_name': 'collectorname',
        'taxon': 'taxon',
        'country/location': 'country_location',
        'iso': 'countrycode',
        'institution': 'institutionname',
        'institution_code': 'institutioncode',
        'year': 'year'
    }
    
    # Create a new dictionary with the expected row names and convert values to strings
    row_data = {mapping[key]: str(value) for key, value in data.items() if key in mapping}
    
    # Create a DataFrame from the dictionary and ensure all data types are string
    df = pd.DataFrame([row_data])
    df = df.astype(str)
    
    return df

# Check if the parameter is not None, not NaN, and not an empty string.
# True if the parameter is valid, False otherwise.
def is_valid(param):
    if param is None:
        return False
    if isinstance(param, str) and param.strip() == "":
        return False
    if isinstance(param, (float, np.float64)) and math.isnan(param):
        return False
    return True

def find_countryname_id(countryname):
    query = f"g.V().has('country','countryname','{countryname}').valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def find_countrycode_id(countrycode):
    query = f"g.V().has('country','iso','{countrycode}').valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def get_country_align(countryname, countrycode):
    result_df = pd.DataFrame()
    # Check if countrycode is not null or NaN
    if is_valid(countrycode):
        result_df = find_countrycode_id(countrycode)
    # Check if countryname is not null or NaN
    if is_valid(countryname):
        temp_df = find_countryname_id(countryname)
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset='iso')
    return result_df

def find_institutionname_id(institutionname):
    query = f"g.V().has('institution','name','{institutionname}').valueMap(true)"
    # The below query is a similar match witch will return all possible institutions containing the name
    # query = f"g.V().has('institution','name',TextP.containing('{institutionname}')).valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def find_institutioncode_id(institutioncode):
    query = f"g.V().has('institution','code','{institutioncode}').valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def get_institution_align(institutionname, institutioncode):
    result_df = pd.DataFrame()
    # Check if countrycode is not null or NaN
    if is_valid(institutioncode):
        result_df = find_institutioncode_id(institutioncode)
    if len(result_df) < 1:
        # Check if countryname is not null or NaN
        if is_valid(institutionname):
            temp_df = find_institutionname_id(institutionname)
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
            result_df = result_df.drop_duplicates(subset='uuid')
    return result_df

def get_taxon_align(taxon):
    parser = TaxonParser(taxon)
    try:
        parsed_name = parser.parse()
        if parsed_name.hasName():
            taxon_name = parsed_name.canonicalNameWithoutAuthorship()
            if parsed_name.hasAuthorship():
                taxon_authorship = parsed_name.authorshipComplete()
                query = f"g.V().has('taxon','name','{taxon_name}').has('taxon', 'authorship', '{taxon_authorship}').valueMap(true)"
                df = wr.neptune.execute_gremlin(client, query)
                if len(df) == 0:
                    query = f"g.V().has('taxon','name','{taxon_name}').valueMap(true)"
                    df = wr.neptune.execute_gremlin(client, query)
            else:
                query = f"g.V().has('taxon','name','{taxon_name}').valueMap(true)"
                df = wr.neptune.execute_gremlin(client, query)
            return df
    except UnparsableNameException as e:
        print("The given taxon info does not seem to be a valid taxon name: \n" + e)

def get_collector_properties(collector_name):
    query = f"""
    g.with('Neptune#ml.endpoint', '{endpoint}').
    with('Neptune#ml.limit', 5).V().hasLabel('collector')
    .properties('authorabbrv_w', 'authorAbbrv_h', 'namelist_w', 'fullname_w', 'fullname1_h', 'fullname2_h', 'fullname_b', 'label_h', 'label_w', 'label_b').
    hasValue(TextP.containing('{collector_name}')).
    with('Neptune#ml.link_prediction')"""
    df = wr.neptune.execute_gremlin(client, query)
    return df

def find_collector(collector_name):
    df = get_collector_properties(collector_name)
    result_df = pd.DataFrame()    
    for index, row in df.iterrows():
        key = row['label']
        query = f"g.V().has('collector', '{key}', TextP.containing('{collector_name}')).valueMap(true)"
        temp_df = wr.neptune.execute_gremlin(client, query)
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset='collectorindex')
    return result_df

# Concatenate names in the list into a single string, separated by spaces, and return the result in the specified format.
def concat_names(name_list):
    filtered_names = [name for name in name_list if name.strip()]
    concatenated_names = ' '.join(filtered_names)
    return f"'{concatenated_names}'"

def get_recordedby_with_collector_info(collector_info):
    query = f"""
    g.with('Neptune#ml.endpoint', '{endpoint}')
      .with('Neptune#ml.limit', 5)
      .V().has('specimen', 'recordedby', '{collector_info}')
      .with('Neptune#ml.link_prediction')
      .outE('recorded_by')
      .inV()
      .valueMap(true)
      .order().by('Neptune#ml.score', desc)
    """
    df = wr.neptune.execute_gremlin(client, query)
    result_df = df.drop_duplicates(subset='collectorindex')
    return result_df

def search_specimen_in_pkb(countrycode,taxon_name,collector_name):
    query = f"""
    g.with('Neptune#ml.endpoint', '{endpoint}').
      V().
      hasLabel('specimen').
      has('countrycode', '{countrycode}').
      has('verbatimscientificname', TextP.containing('{taxon_name}')).
      has('recordedby', TextP.containing('{collector_name}')).
      valueMap(true)
    """
    df = wr.neptune.execute_gremlin(client, query)
    return df

expected_collector_columns = ['collectorindex','authorabbrv_w','authorAbbrv_h','wikiid','wikidata_b',
                              'harvardindex_w_merged','harvardindex_w','harvardindex_w_wh','harvardindex',
                              'orcid_b','bioid','bionomia_w']
def align_collector(name_list):
    result_df = pd.DataFrame(columns=expected_collector_columns)    
    for name in name_list:
        if is_valid(name):
            temp_df = find_collector(name)
            # Add missing columns to temp_df
            for col in expected_collector_columns:
                if col not in temp_df.columns:
                    temp_df[col] = pd.NA
            # Ensure temp_df has the columns in the same order as expected_columns
            temp_df = temp_df[expected_collector_columns]
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
    result_df = result_df.drop_duplicates(subset='collectorindex')
    return result_df

def get_unique_collector_indices(df1, df2):
    # Extract the collectorindex from both DataFrames
    recordby_collector_indices = recordby_df['collectorindex'].tolist()
    collector_collector_indices = collector_df['collectorindex'].tolist()
    # Combine the lists without duplication
    combined_collector_indices = recordby_collector_indices + collector_collector_indices
    unique_collector_indices = list(set(combined_collector_indices))
    return unique_collector_indices

def get_recommend_collector(collectorindex_list):
    result_df = pd.DataFrame()
    for collectorindex in collectorindex_list:
        query = f"g.V().has('collector','collectorindex','{collectorindex}').valueMap(true)"
        df = wr.neptune.execute_gremlin(client, query)
        result_df = pd.concat([result_df, df], ignore_index=True)
    return result_df

def print_collector_recommended(df):
    for index, row in df.iterrows():
        st.text(f"The {index + 1} recommended collector:")
        
        # Print collector index, assuming this column always exists
        st.text("Collector Index: " + str(row['collectorindex']))
        
        # Check and print Author Abbreviation
        author_abbrv = "; ".join(filter(pd.notna, [
            row.get('authorabbrv_w', None), 
            row.get('authorAbbrv_h', None)
        ]))
        if author_abbrv:
            st.text("Author Abbreviation: " + author_abbrv)
        
        # Check and print Wikipedia IDs
        wiki_id = "; ".join(filter(pd.notna, [
            row.get('wikiid', None), 
            row.get('wikidata_b', None)
        ]))
        if wiki_id:
            st.text("Wikipedia IDs: " + wiki_id)
        
        # Check and print Harvard Index IDs
        harvard_index = "; ".join(filter(pd.notna, [
            row.get('harvardindex_w_merged', None), 
            row.get('harvardindex_w', None),
            row.get('harvardindex_w_wh', None),
            row.get('harvardindex', None)
        ]))
        if harvard_index:
            st.text("Harvard Index IDs: " + harvard_index)
        
        # Check and print Orcid IDs
        orcid_id = row.get('orcid_b', None)
        if pd.notna(orcid_id):
            st.text("Orcid IDs: " + str(orcid_id))
        
        # Check and print Bionomia IDs
        bionomia_id = "; ".join(filter(pd.notna, [
            row.get('bioid', None), 
            row.get('bionomia_w', None)
        ]))
        if bionomia_id:
            st.text("Bionomia IDs: " + bionomia_id)
        
        st.text("================================================")

def remove_comma(input_string):
    return input_string.replace(",", "")

##############################################################################
##############################################################################
##############################################################################


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    base64_image = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{base64_image}"
    
    # Convert image to text 
    response = gpt_client.chat.completions.create(
        model='gpt-4o-mini', 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": promt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ],
            }
        ],
        max_tokens=500,
    )

    string_result = response.choices[0].message.content
    # Display the extracted text
    st.subheader("Extracted Text:")
    st.text(string_result)
    result_df = convert_to_dataframe(string_result)
    # Streamlit method to display dataframe
    st.dataframe(result_df)
    
    namelist= result_df['collectorname'].tolist()
    countryname = result_df['country_location'].tolist()[0]
    countrycode = result_df['countrycode'].tolist()[0]
    institutionname = result_df['institutionname'].tolist()[0]
    institutioncode = result_df['institutioncode'].tolist()[0]
    taxon = result_df['taxon'].tolist()[0]
    
    st.subheader("Aligning the given information to the PKB:")
    st.text("Aligning Country...")
    country_df = get_country_align(countryname,countrycode)
    st.dataframe(country_df)
    
    st.text("Aligning Institution...")
    institution_df = get_institution_align(institutionname, institutioncode)
    st.dataframe(institution_df)
    
    st.text("Aligning/Suggesting Taxon...")
    taxon_df = get_taxon_align(remove_comma(taxon))
    st.dataframe(taxon_df)
    
    st.text("Aligning/Suggesting Collector...")
    if len(namelist) >= 1:
        if not is_valid(namelist[0]):
            st.text("There's no collector information extracted from GPT model.")
        else:
            collector_df = align_collector(namelist)
            st.dataframe(collector_df)
    
    st.text("Trying to find possible matched GBIF records...")
    if not is_valid(namelist[0]):
        st.text("Collector information is missing from GPT model. Suggesting records requires country, taxon and collector information.")
    else:
        in_gbif_df = search_specimen_in_pkb(countrycode,taxon,namelist[0])
        st.dataframe(in_gbif_df)
    
    st.text("Predicting the highest possible recordBy...")
    if len(namelist) >= 1:
        if not is_valid(namelist[0]):
            st.text("There's no collector information extracted from GPT model.")
        else:
            if len(namelist) >= 2:
                nameinfo = concat_names(namelist)
                recordby_df = get_recordedby_with_collector_info(nameinfo)
            else:
                recordby_df = get_recordedby_with_collector_info(namelist[0])
            st.dataframe(recordby_df)
    
            # Ensure collector_df columns are a subset of recordby_df columns
            if len(recordby_df) >= 1:
                collectorindex_list = get_unique_collector_indices(recordby_df, collector_df)
            else:
                collectorindex_list = collector_df['collectorindex'].tolist()
    
            recommend_collector_df = get_recommend_collector(collectorindex_list)
            print_collector_recommended(recommend_collector_df)
    
    
##############################################################################
##############################################################################
##############################################################################
    
    
    # Create the graph network
    g = Network(height='400px', width='80%', heading='')

    no_node = 0
    # Add nodes with different colors
    # Plot the current specimen node
    g.add_node(no_node, label='current specimen', color='#f7b5ca')
    no_node += 1
    
    # Plot the current specimen node
    if len(country_df) >= 1:
        g.add_node(no_node, label=country_df['iso'][0], color='#f5c669')
        # Add edges
        g.add_edge(0, no_node, color='black', label='origing_country')
        no_node += 1
        
    if len(institution_df) >= 1:
        if len(institution_df) == 1:
            g.add_node(no_node, label=institution_df['name'][0], color='#82b6fa')
            g.add_edge(0, no_node, color='black', label='discovering_institution')
            no_node += 1
            g.add_node(no_node, label=institution_df['country'][0], color='#f5c669')
            g.add_edge(no_node-1, no_node, color='black', label='located_in')
            no_node += 1
        if len(institution_df) > 1:
            # Only plot the highest five institution
            inst_list = institution_df['name'].tolist()[:5]
            cty_list = institution_df['country'].tolist()[:5]
            for inst, cty in zip(inst_list, cty_list):
                g.add_node(no_node, label=inst, color='#82b6fa')
                g.add_edge(0, no_node, color='black', label='discovering_institution', dashes=True)
                no_node += 1
        
                g.add_node(no_node, label=cty, color='#f5c669')
                g.add_edge(no_node-1, no_node, color='black', label='located_in')
                no_node += 1

    num_taxon = no_node
    if len(taxon_df) > 1:
        taxonlist = (taxon_df['name']+' '+taxon_df['authorship']).tolist()
        for taxon in taxonlist:
            g.add_node(num_taxon, label=taxon, color='#befa82')
            g.add_edge(0, num_taxon, color='black', label='determination', dashes=True)
            num_taxon += 1
    elif len(taxon_df) == 1:
        taxon = taxon_df['name'].iloc[0] + ' ' + taxon_df['authorship'].iloc[0]
        g.add_node(num_taxon, label=taxon, color='#befa82')
        g.add_edge(0, num_taxon, color='black', label='determination')
        num_taxon += 1
    else:
        pass

    # dashes represent suggestions
    num_name = num_taxon
    for name in namelist:
        g.add_node(num_name, label=name, color='#fad5c0')
        g.add_edge(0, num_name, color='black', label='recorded_by', dashes=True)
        num_name = num_name+1

    # Generate and show the network
    html_file = 'example.html'
    g.save_graph(html_file)
    
    st.subheader("Displaying sub-graph of the given specimen:")
    HtmlFile = open("example.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 900,width=900)
    
    no_node = 0
