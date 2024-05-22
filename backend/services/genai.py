import docstring_parser
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_vertexai import VertexAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from vertexai.generative_models import GenerativeModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'


class Flashcard(BaseModel):
    concept: str = Field(description="concept")
    definition: str = Field(description="definition")

class GenAIProcessor:

    def __init__(self,model_name,project):
        self.model = VertexAI(model_name=model_name,project=project)

    def generate_document_summary(self,documents :list,**args):
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(
            llm = self.model,
            chain_type = chain_type,
            **args
        )
        return chain.run(documents)
    
    def count_billable_tokens(self, docs : list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0

        logger.info("Counting total billable characters...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total
    
    def get_model(self):
        return self.model

class YoutubeProcessor:
    #Retrieve the full transcript

    def __init__(self, genai_processor: GenAIProcessor):
        #Creates a Character Text Splitter that splits text documents into chunks of characters.
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 0
        )
        self.genAIProcessor = genai_processor

    #Retrieves the documents and splits it, returning the metadata as well as the transcript for the youtube video
    def retrieve_youtube_documents(self, video_url: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(video_url,add_video_info = True)
        docs = loader.load()
        
        result = self.text_splitter.split_documents(docs)
        
        
        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.genAIProcessor.count_billable_tokens(result)

        if verbose:

            logger.info(f"{author}\n{length}\n{title}\n{total_size}")

            
        
        return result
    def find_key_concepts(self, documents:list, sample_size: int=0, verbose = False):
        #iterate through all documents of group size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group Size is larger than the number of documents")
        

        #Optimize sample size given no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose: logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample Size: {sample_size}")

        #Find the number of documents in each group
        num_docs_per_group = len(documents)//sample_size+(len(documents)%sample_size >0)

      
        #Check thresholds for response quality
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and the output quality will be downgraded significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more thant 5 documents and output quality will likely to be degraded. Consider increasing sample size.")
        
        #Split the  document into chunk of size num_docs per game.
        group = [documents[i:i+num_docs_per_group] for i in range(0, len(documents),num_docs_per_group)]
        
        batch_concepts = []
        batch_cost = 0

        logger.info("finding key concepts...")

        for group in tqdm(group):
            #Combine content of documents per group

            group_content = ""

            for doc in group:
                group_content += doc.page_content
            
            parser = JsonOutputParser(pydantic_object=Flashcard)

            #Prompt for finding templates
            prompt = PromptTemplate(
                template = """
                Find and define key terms and definitions found in the text:
                {text}

                Respond in the following format as a JSON object without any backticks separating each term with commas:
                {{
                    {{"term" :
                    "definition}}
                    
                    {{"term" :
                    "definition"}}

                    {{"term" :
                    "definition"}}

                    {{"term" :
                    "definition"}}

                    {{"term" :
                    "definition"}}
                
                ...}}
                """,
                input_variables=["text"]
            )


            #Create chain
            chain = prompt | self.genAIProcessor.model | parser

            #Run chain
            output_concept = chain.invoke({"text": group_content})
            batch_concepts.append(output_concept)

            logging.info(print(concept.term) for concept in batch_concepts)
            #Post Processing Observation
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125

                logging.info(f"Running chain on {len(group)} documents...")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total cost: {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000)*0.000375

                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total cost: {total_output_cost}")

                batch_cost += total_input_cost + total_output_cost
                logging.info(f"Total group cost: {total_input_cost + total_output_cost}\n")
        
        #Convert json JSON string from batch concepts to Python Dict
        processed_concepts = []
        for concept in batch_concepts:
            concept_converted = json.dumps(concept)
            processed_concepts.append(json.loads(concept_converted))

        logging.info(f"Total Analysis Cost: ${batch_cost}")
        return processed_concepts