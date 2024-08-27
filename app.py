"""
summarise any webpage (url) and prompt based on the content using a locally run LLM.
"""

import requests
import re
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama.llms import OllamaLLM

import torch
import ollama
import dash

from dash import html
from dash import dcc
from dash.dependencies import Output, Input


def get_text_from_url(url):
    """
    extract text from anywebpage url
    """
    page = requests.get(url)
    # extract and clean text
    soup = BeautifulSoup(page.text, "html.parser")
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
        
    return text


def get_text_chunks(text):
    """
    split the raw text into chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks


def get_text_embeddings(text_chunks, embedding_model='mxbai-embed-large'):
    """
    create embeddings locally
    """
    text_embeddings = []
    for context in text_chunks:
        response = ollama.embeddings(model=embedding_model, prompt=context)
        text_embeddings.append(response["embedding"])

    return text_embeddings


def get_context_texts(prompt_text, extracted_text_embeddings, extracted_text_chunks, top_k=3, embedding_model='mxbai-embed-large'):
    """
    perform similarity search between prompt text and embedding store
    """
    input_embedding = ollama.embeddings(model=embedding_model, prompt=prompt_text)["embedding"]
    input_embedding = torch.tensor(input_embedding).unsqueeze(0)
    # compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(input_embedding, extracted_text_embeddings)
    # adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # get the corresponding context from the extracted text chunks
    context_texts_list = [extracted_text_chunks[idx].strip() for idx in top_indices]

    return context_texts_list


def prompt_llm_with_context(user_prompt, context, model='llama3'):
    """
    get output from llm with user_prompt and extracted context
    """
    prompt = ChatPromptTemplate([
        ("system", "You are a helpful AI bot. You are supposed to give accurate answers which are inline with the CONTEXT and QUESTION. DON'T BE VERBOSE."),
        ("human", "{prompt_with_context}"),
    ])

    model = OllamaLLM(model=model)
    # create chain
    chain = prompt | model

    prompt_with_context = f"""
    CONTEXT(from url): {context}
    QUESTION: {user_prompt}
    """

    response = chain.invoke({"prompt_with_context": f"{prompt_with_context}"})

    return response


# ======================== SETUP DASH APP ======================== 

# Initialize the Dash app
app = dash.Dash()

# Define the layout of the app
app.layout = html.Div(
    style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'height': '100vh',
        'flex-direction': 'column',
        'text-align': 'center'
    },
    children=[
        html.H1(children='Ask LLM based on context from URL', style={'margin-bottom': '30px'}),
        
        html.Div([
            html.Label('Choose local LLM:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='llm-model-dropdown',
                options=[
                    {'label': 'llama3', 'value': 'llama3'},
                    {'label': 'mistral', 'value': 'mistral'}
                ],
                value='llama3',  # Default value
                style={'width': '300px', 'margin-bottom': '20px'}
            )
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Label('Choose embedding model:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='embedding-model-dropdown',
                options=[
                    {'label': 'mxbai-embed-large', 'value': 'mxbai-embed-large'},
                    {'label': 'nomic-embed-text', 'value': 'nomic-embed-text'}
                ],
                value='mxbai-embed-large',  # Default value
                style={'width': '300px', 'margin-bottom': '20px'}
            )
        ], style={'margin-bottom': '20px'}),
        
        dcc.Input(id='input-url', type='text', debounce=True, placeholder='Paste the URL for context', style={'width': '400px', 'margin-bottom': '20px'}),
        dcc.Input(id='input-question', type='text', debounce=True, placeholder='Ask a question', style={'width': '400px', 'margin-bottom': '30px'}),
        
        dcc.Loading(
            id="loading",
            type="default",
            children=html.Div(
                id='answer-div',
                children='',
                style={
                    'padding': '10px',
                    'backgroundColor': '#f0f0f0',
                    'border-radius': '10px',
                    'width': '60%',
                    'textAlign': 'center',
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'margin': '0 auto', 
                    'min-height': '50px'  
                    }
            )
        )
    ]
)

@app.callback(
    Output(component_id='answer-div', component_property='children'),
    Input(component_id='llm-model-dropdown', component_property='value'),
    Input(component_id='embedding-model-dropdown', component_property='value'),
    Input(component_id='input-url', component_property='value'),
    Input(component_id='input-question', component_property='value'),
    prevent_initial_call=True
)
def web_interface_io(llm_model, embedding_model, url, question):
    extracted_text = get_text_from_url(url)
    extracted_text_chunks = get_text_chunks(extracted_text)
    extracted_text_embeddings = get_text_embeddings(extracted_text_chunks, embedding_model=embedding_model)
    extracted_text_embeddings = torch.tensor(extracted_text_embeddings)
    context_texts_list = get_context_texts(question, extracted_text_embeddings, extracted_text_chunks, top_k=3, embedding_model=embedding_model)

    full_context = ''
    for index, context in enumerate(context_texts_list):
        context = f'{index+1}. {context} \n'
        full_context += context

    llm_out = prompt_llm_with_context(question, full_context, model=llm_model)
    return llm_out


# Run the web app
if __name__ == '__main__':
    app.run_server(debug=False)
