from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_vertexai import VertexAI
from langchain.chains.summarize import load_summarize_chain
import logging


class YoutubeProcessor:
    #Retrieve the full transcript

    def __init__(self):
        #Creates a Character Text Splitter that splits text documents into chunks of characters.
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 0
        )

    #Retrieves the documents and splits it, returning the metadata as well as the transcript for the youtube video
    def retrieve_youtube_documents(self, video_url: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(video_url,add_video_info = True)
        docs = loader.load()
        
        result = self.text_splitter.split_documents(docs)
        
        
        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)

        if verbose:
            print(f"{author}\n{length}\n{title}\n{total_size}")
        
        return result