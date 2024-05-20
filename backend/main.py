from click import group
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

from services.genai import (YoutubeProcessor,
                            GenAIProcessor)

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl
    # advanced settings

app = FastAPI()

genai_processor = GenAIProcessor("gemini-pro","quizify-radical-ai")

#Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    #Doing the analysis
        
        processor = YoutubeProcessor(genai_processor=genai_processor)
        result = processor.retrieve_youtube_documents(str(request.youtube_link),verbose=True)

        #Find summary
        summary = genai_processor.generate_document_summary(result)

        #Find key concepts
        key_concepts = processor.find_key_concepts(result,verbose = True)
        return{
            "key concepts" : key_concepts
        }


@app.get("/health")
def health():
    return {"status": "ok"}