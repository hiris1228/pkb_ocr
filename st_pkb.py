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

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    #"[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    #"[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("PKB OCR")

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

promt1 = "Find collector name, taxon, country/location(convert into ISO Alpha-2 country that only has 252 Country), institution(convert into institution code) and year. Only return content and their classes in the below format: collector name, taxon, country/location, ISO, institution, institution code, year"

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
    if pd.notnull(countrycode):
        result_df = find_countrycode_id(countrycode)
    # Check if countryname is not null or NaN
    if pd.notnull(countryname):
        temp_df = find_countryname_id(countryname)
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset='iso')
    return result_df

def find_institutionname_id(institutionname):
    query = f"g.V().has('institution','name','{institutionname}').valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def find_institutioncode_id(institutioncode):
    query = f"g.V().has('institution','code','{institutioncode}').valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

def get_institution_align(institutionname, institutioncode):
    result_df = pd.DataFrame()
    # Check if countrycode is not null or NaN
    if pd.notnull(institutioncode):
        result_df = find_institutionname_id(institutionname)
    # Check if countryname is not null or NaN
    if pd.notnull(institutionname):
        temp_df = find_institutioncode_id(institutioncode)
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset='uuid')
    return result_df

def find_taxon_id(taxon):
    query = f"g.V().has('taxon','name',TextP.containing('{taxon}')).valueMap(true)"
    df = wr.neptune.execute_gremlin(client, query)
    return df

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
    taxon_df = find_taxon_id(taxon)
    st.dataframe(taxon_df)
    
    st.text("Aligning/Suggesting Collector...")
    collector_df = align_collector(namelist)
    st.dataframe(collector_df)
    
    st.text("Trying to find matched gbif records...")
    in_gbif_df = search_specimen_in_pkb(countrycode,taxon,namelist[0])
    st.dataframe(in_gbif_df)
    
    st.text("Predicting the highest possible recordBy...")
    recordby_df = get_recordedby_with_collector_info(namelist[0])
    st.dataframe(recordby_df)
    
    # Create the network
    g = Network(height='400px', width='80%', heading='')

    # Add nodes with different colors
    g.add_node(0, label='current specimen', color='#f7b5ca')
    g.add_node(1, label=countrycode, color='#f5c669')
    g.add_node(2, label=institutionname, color='#82b6fa')

    # Add edges
    g.add_edge(0, 1, color='black', label='origing_country')
    g.add_edge(0, 2, color='black', label='discovering_institution')

    num_taxon = 3
    if len(taxon_df) >= 1:
        taxonlist = (taxon_df['name']+' '+taxon_df['authorship']+' id_'+taxon_df['taxonid']).tolist()
        for taxon in taxonlist:
            g.add_node(num_taxon, label=taxon, color='#befa82')
            g.add_edge(0, num_taxon, color='black', label='determination', dashes=True)
            num_taxon = num_taxon+1
    else:
        g.add_node(num_taxon, label=taxon, color='#befa82')
        g.add_edge(0, num_taxon, color='black', label='determination', dashes=True)
        num_taxon = num_taxon+1

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
