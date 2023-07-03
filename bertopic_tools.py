import os
import torch
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_pages import add_page_title
from typing import Any, Optional, Tuple, Union

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI


class BertopicTools:
    """
    BertopicTools is a class that facilitates the creation, loading and analysis of topic models using BERTopic.

    Attributes:
        model_name (str): Model name for the OpenAI model.
        AGENTS (dict): Dictionary to manage 'topic_info' queries.

    Methods:
        __init__(model_name): Initializes the class with a given model name.
        agency(df, temperature, verbose, max_iterations): Creates a Pandas DataFrame agent with an OpenAI model.
        agent_runner(agent, query): Runs a query through a given agent.
        save_model(model, save_path): Saves a given model to a specified location.
        fit_transform(_df, column, save_path, embedding_model_name, language, top_n_words, n_gram_range, min_topic_size, nr_topics, calculate_probabilities): Creates a topic model and fits it to a specified column of a DataFrame.
        load_transform(_df, column, model_path, calculate_probs, embedding_model_name): Loads an existing model and transforms a specified column of a DataFrame.
        display_data(): Displays the results of topic modeling using various plots.
        render(): Executes a Streamlit application to interactively work with BERTopic.
    """
    DOCS = 'docs'
    TOPIC_MODEL = 'topic_model'
    EMBEDDINGS = 'embeddings'
    TOPIC_MODELING_RESULT = 'topic_modeling_result'
    TOPICS = 'topics'
    PROBS = 'probs'
    DATA = 'data'
    LANGCHAIN_HISTORY = 'langchain_history'
    AGENTS = {
        'topic_info': []
    }

    def __init__(self, model_name: str = 'text-davinci-003'):
        """
        Initializes the BertopicTools class with a given model name.
        
        Args:
            model_name (str): Model name for the OpenAI model.
        """
        self.model_name = model_name
        load_dotenv()  # Load environment variables from .env file
        os.environ.setdefault('OPENAI_API_KEY', 'default_value_if_not_found')

    def agency(self, df: pd.DataFrame, temperature: float = 0, verbose: bool = True, max_iterations: int = 5) -> Any:
        """
        Creates a Pandas DataFrame agent with an OpenAI model.

        Args:
            df (pd.DataFrame): Input dataframe to the agent.
            temperature (float): Temperature parameter for the OpenAI model.
            verbose (bool): If True, prints verbose output.
            max_iterations (int): Maximum number of iterations for the agent.

        Returns:
            Any: The created agent.
        """
        llm = OpenAI(
            temperature=temperature,
            model_name=self.model_name,
            verbose=verbose
        )
        agent = create_pandas_dataframe_agent(llm, df, verbose=verbose, max_iterations=max_iterations)
        return agent

    def agent_runner(self, agent: Any, query: str) -> Any:
        """
        Runs a query through a given agent.

        Args:
            agent (Any): The agent to run the query.
            query (str): The query to run.

        Returns:
            Any: The result of the query.
        """
        return agent.run(query)

    def save_model(self, model: Any, save_path: str) -> None:
        """
        Runs a query through a given agent.

        Args:
            agent (Any): The agent to run the query.
            query (str): The query to run.

        Returns:
            Any: The result of the query.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f'{save_path}BERT-{timestamp}'
        model.save(model_path)

    def fit_transform(self, _df: pd.DataFrame, column: str, save_path: str = '', embedding_model_name: Union[str, None] = None, 
                      language: str = '', top_n_words: int = 10, n_gram_range: Tuple[int, int] = (1, 3), 
                      min_topic_size: int = None, nr_topics: int = None, calculate_probabilities: bool = False
                      ) -> Tuple[pd.DataFrame, BERTopic, list, list, bool]:
        """
        Fits a BERTopic model to a column of a DataFrame and transforms it into topic representations.

        Args:
            _df (pd.DataFrame): Input DataFrame.
            column (str): Column of the DataFrame to fit the model to.
            save_path (str): Location to save the created model. Default is empty string, which doesn't save the model.
            embedding_model_name (Union[str, None]): Name of the transformer model for embeddings.
            language (str): Language of the documents. Default is '' (English).
            top_n_words (int): The number of words per topic to display in topic representation. Default is 10.
            n_gram_range (Tuple[int, int]): Range of n-gram values to use. Default is (1, 3).
            min_topic_size (int): Minimum size of the topic. Default is None.
            nr_topics (int): The number of topics. Default is None.
            calculate_probabilities (bool): Whether to calculate probabilities. Default is False.

        Returns:
            Tuple[pd.DataFrame, BERTopic, list, list, bool]: Tuple consisting of DataFrame with topic representations,
                BERTopic model, embeddings, topics, probabilities, and a flag indicating whether the model was saved.
        """
        _df = _df.reset_index(drop=True)
        docs = _df[column].tolist()
        topic_model, topics, probs, embeddings, model_saved = None, None, None, None, False
        if embedding_model_name:
            sentence_model = SentenceTransformer(embedding_model_name)
            embeddings = sentence_model.encode(docs, show_progress_bar=True)
            topic_model = BERTopic(language=language, top_n_words=top_n_words, n_gram_range=n_gram_range,
                                   min_topic_size=min_topic_size, nr_topics=nr_topics, calculate_probabilities=calculate_probabilities, 
                                   embedding_model=sentence_model)
            topics, probs = topic_model.fit_transform(docs, embeddings)
        else:
            topic_model = BERTopic(language=language, top_n_words=top_n_words, n_gram_range=n_gram_range,
                                   min_topic_size=min_topic_size, nr_topics=nr_topics, calculate_probabilities=calculate_probabilities)
            topics, probs = topic_model.fit_transform(docs)
            embeddings = topic_model.topic_embeddings_

        if len(save_path.strip()) > 0:
            self.save_model(topic_model, save_path)
            model_saved = True
    
        # topic_df = pd.DataFrame({"Document": docs, "Topic": topics})
        # topic_info = topic_model.get_topic_info()
        # topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
        # topic_df['Topic Name'] = topic_df['Topic'].map(topic_name_map)

        # return topic_df, topic_model, embeddings, topics, probs, model_saved
        _df['Document'] = docs
        _df['Topic'] = topics
        topic_info = topic_model.get_topic_info()
        topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
        _df['Topic Name'] = _df['Topic'].map(topic_name_map)

        return _df, topic_model, embeddings, topics, probs, model_saved

    def load_transform(self, _df: pd.DataFrame, column: str, model_path: str = "", 
                          calculate_probs: bool = True, 
                          embedding_model_name: Optional[str] = None
                          ) -> Tuple[pd.DataFrame, BERTopic, list, Optional[list]]:
        """
        Loads a BERTopic model and transforms a column of a DataFrame into topic representations.

        Args:
            _df (pd.DataFrame): Input DataFrame.
            column (str): Column of the DataFrame to fit the model to.
            model_path (str): Location to load the saved model. Default is an empty string, which does not load a model.
            calculate_probs (bool): Whether to calculate probabilities. Default is True.
            embedding_model_name (Optional[str]): Name of the transformer model for embeddings. Default is None.

        Returns:
            Tuple[pd.DataFrame, BERTopic, list, Optional[list]]: Tuple consisting of DataFrame with topic representations,
                BERTopic model, topic embeddings, topics, and probabilities.
        """
        _df = _df.reset_index(drop=True)
        docs = _df[column].tolist()
        model_path = model_path.strip()
        topic_model = BERTopic.load(model_path)
        probs = None
        if embedding_model_name:
            sentence_model = SentenceTransformer(embedding_model_name)
            if sentence_model.tokenizer.pad_token is None: 
                sentence_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            embeddings = sentence_model.encode(docs, show_progress_bar=True)
            if calculate_probs: 
                topics, probs = topic_model.transform(docs, embeddings)
            else:
                topics = topic_model.get_topics()
        else:
            if calculate_probs: 
                topics, probs = topic_model.transform(docs)
            else:
                topics = topic_model.get_topics()

        # topic_df = pd.DataFrame({"Document": docs, "Topic": topics})
        # topic_info = topic_model.get_topic_info()
        # topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
        # topic_df['Topic Name'] = topic_df['Topic'].map(topic_name_map)

        # return topic_df, topic_model, topic_model.topic_embeddings_, topics, probs
        _df['Document'] = docs
        _df['Topic'] = topics
        topic_info = topic_model.get_topic_info()
        topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
        _df['Topic Name'] = _df['Topic'].map(topic_name_map)

        return _df, topic_model, topic_model.topic_embeddings_, topics, probs
    
    def display_data(self) -> None:
      """
      Visualizes the results of topic modeling using various plots, including a topic hierarchy, intertopic distance map, 
      top words score, similarity matrix, and term score decline per topic.
      Additionally, allows the user to view the probabilities distribution for a specific document.
      Note: This function uses the 'streamlit' library for creating the web-based user interface, and thus does not have return values.
      """
      docs = st.session_state[self.DOCS]
      topic_model = st.session_state[self.TOPIC_MODEL]
      topics = st.session_state[self.TOPICS]
      probs = st.session_state[self.PROBS]
      
      topic_info = topic_model.get_topic_info()
      with st.expander("Main view"):
           # Extract unique topics
          topic_names = docs['Topic Name'].unique().tolist()
          selected_topics = st.multiselect('Select Topics', options=topic_names)

          # Control Number of Documents
          num_docs = st.slider('Number of Documents', min_value=1, max_value=50, value=5)
          for topic in selected_topics:
              st.write("---")
              st.write(f"**{topic}**")
              
              # Filter dataframe for the specific topic

              topic_df = docs[docs['Topic Name'] == topic]

              # Limit to the number of documents
              num_docs_topic = min(num_docs, len(topic_df))
              #st.session_state[self.TOPIC_MODELING_RESULT] = topic_df
              topic_df = topic_df.sample(n=num_docs_topic)

              for index, row in topic_df.iterrows():
                  st.write(f"- {row['Document']}")
                  
      with st.expander("Documents"):
          st.dataframe(docs, use_container_width=True)

      with st.expander("Topics"):
          st.dataframe(topics, use_container_width=True)

      with st.expander("Topic hierarchy"):
          try:
              st.plotly_chart(topic_model.visualize_hierarchy(top_n_topics=50, custom_labels=True), use_container_width=True, theme="streamlit")
          except:
              st.error('Could not plot, check topics')


      with st.expander("Topic info"):
          ti_col1, ti_col2 = st.columns(2, gap='small')
          topic_info_agent = self.agency(topic_info)
          query = ti_col2.text_input('Ask AI', help='Ask AI about this information using lang chain. Will run LLM generated code')
          if ti_col2.button('Question'):
              response = self.agent_runner(topic_info_agent, query)
              st.session_state[self.LANGCHAIN_HISTORY]['topic_info'].append({'query':query,'response':response})
          for qa in st.session_state[self.LANGCHAIN_HISTORY]['topic_info']:
              ti_col2.write('Q: {}'.format(qa['query']))
              ti_col2.write('A: {}'.format(qa['response']))
          try:
              ti_col1.dataframe(topic_info, use_container_width=True)
          except:
              st.error('Could not plot, check topics')

      with st.expander("Intertopic distance map"):
          try:
              st.plotly_chart(topic_model.visualize_topics(), use_container_width=True, theme="streamlit")
          except:
              st.error('Could not plot, check topics')

      with st.expander("Top words score"):
          try:
              st.plotly_chart(topic_model.visualize_barchart(top_n_topics=15, custom_labels=True), use_container_width=True, theme="streamlit")
          except:
              st.error('Could not plot, check topics')

      with st.expander("Similarity matrix"):
          try:
              st.plotly_chart(topic_model.visualize_heatmap(n_clusters=20, width=800, height=800, custom_labels=True), use_container_width=True, theme="streamlit")
          except:
              st.error('Could not plot, check topics')


      with st.expander("Term score decline per topic"):
          try:
              st.plotly_chart(topic_model.visualize_term_rank(custom_labels=True), use_container_width=True, theme="streamlit")
          except:
              st.error('Could not plot, check topics')

      # with st.expander("Probabilities distribuition on specific doc"):
      #     view_doc = st.number_input('View doc', 0, len(st.session_state[DOCS]) - 1)
      #     col1, col2 = st.columns(2, gap='small')
      #     if probs is not None:
      #         try:
      #             col1.plotly_chart(topic_model.visualize_distribution(probs[view_doc], min_probability=0.015, custom_labels=True), use_container_width=True, theme="streamlit")
      #         except:
      #             st.error('Could not plot, check topics')
      #     else:
      #         col1.write("Probabilities were not calculated, you must transform the documents to obtain them.")
      #     col2.write(st.session_state[DOCS][view_doc])
    def render(self):
        """
        Controls the user interface for topic modeling with the BERTopic model in a Streamlit application.

        The function enables the user to select the dataset column to use in the pipeline, 
        choose the transformer model for custom embeddings, specify parameters for the BERTopic model, 
        and train a new model or load an existing one.

        The method also displays the loaded data and output of the topic modeling, 
        and saves the generated documents, topic model, embeddings, topics, and probabilities in the session state.

        Note:
            The method does not return anything but modifies the Streamlit session state.

        Raises:
            st.error: If no document is uploaded.
            st.warning: If CUDA is not available and the application might run slower.
        """
        try:
            st.set_page_config(layout="wide")
        except:
            pass
        add_page_title()

        if self.DATA not in st.session_state:
            st.error("It seems like no document is uploaded, you should probably visit the Homepage")
            return
        
        if not torch.cuda.is_available():
            st.warning('Model is not running on CUDA. This might slow down the application.')
        
        data = st.session_state[self.DATA]
        string_columns = [column for column in data.columns if data[column].dtype == 'object']
        column = st.sidebar.selectbox('Select the column you want to use on this pipeline', string_columns, len(string_columns) -1)

        sample_empty = st.empty()

        st.session_state[self.DOCS] = st.session_state.get(self.DOCS, None)
        st.session_state[self.TOPIC_MODEL] = st.session_state.get(self.TOPIC_MODEL, None)
        st.session_state[self.TOPICS] = st.session_state.get(self.TOPICS, None)
        st.session_state[self.PROBS] = st.session_state.get(self.PROBS, None)
        st.session_state[self.LANGCHAIN_HISTORY] = st.session_state.get(self.LANGCHAIN_HISTORY, self.AGENTS)
        
        if st.sidebar.radio("Custom embeddings",["On", "Off"], index=1) == "On":
            embedding_model_name = st.sidebar.text_input("Embedding model name", help="Enter the name of the transformer model you want to use, leaving empty defaults to BERTopic's embeddings.")
        else:
            embedding_model_name=''
        if st.sidebar.radio("Choose",["New model", "Load model"]) == "New model":
            st.sidebar.header("BERTopic Parameters")
            output_path = st.sidebar.text_input("Output path", help="Leaving empty will not save the model")
            language = st.sidebar.selectbox('Model language', ['english', 'multilingual', 'portuguese', 'french', 'german'])
            top_n_words = st.sidebar.slider("Top n words", 0, 100, 10)
            n_gram_range = (st.sidebar.slider("N-gram range start", 1, 5, 1), st.sidebar.slider("N-gram range end", 1, 5, 3))
            min_topic_size = st.sidebar.slider("Minimum Topic Size", 0, 100, 10)
            nr_topics = st.sidebar.slider("Number of Topics", 0, 100, 0)
            if st.sidebar.button('Train'):
                with st.spinner('transforming documents...'):
                    docs, topic_model, embeddings, topics, probs, model_saved = self.fit_transform(
                        _df = data, column=column, save_path=output_path, 
                        embedding_model_name=embedding_model_name, language=language, top_n_words=top_n_words, 
                        n_gram_range=n_gram_range, min_topic_size=min_topic_size, nr_topics= None if nr_topics == 0 else nr_topics
                        )
                    st.success("Training Completed!")
                    st.session_state[self.DOCS] = docs
                    st.session_state[self.TOPIC_MODEL] = topic_model
                    st.session_state[self.EMBEDDINGS] = embeddings
                    st.session_state[self.TOPICS] = topics
                    st.session_state[self.PROBS] = probs

                    if model_saved:
                        st.success("Model saved")
                    else:
                        st.error("Model not saved")

        else:
            calculate_probs = st.sidebar.radio("Calculate probs",[True, False])
            model_path = st.sidebar.text_input("Model path", help="Required")
            if st.sidebar.button('Load'):
                with st.spinner('transforming documents...'):
                    docs, topic_model, embeddings, topics, probs = self.load_transform(_df=data, column=column, model_path=model_path, calculate_probs=calculate_probs, embedding_model_name=embedding_model_name)
                    st.session_state[self.DOCS] = docs
                    st.session_state[self.TOPIC_MODEL] = topic_model
                    st.session_state[self.EMBEDDINGS] = embeddings
                    st.session_state[self.TOPICS] = topics
                    st.session_state[self.PROBS] = probs

        if st.session_state[self.TOPIC_MODEL] is not None and st.session_state[self.TOPICS] is not None:
            self.display_data()
            with sample_empty.expander("Current state sample out of {}".format(len(data))):
                data = st.session_state[self.DATA]
                min_value = min(1, len(data))
                max_value = max(50, len(data))
                sample_size = st.slider('Sample size', min_value=min_value, max_value=max_value, value=int(max_value/2))
                st.dataframe(data.sample(n=sample_size), use_container_width=True)
        
if __name__ == "__main__":
    BertopicTools().render()
