
import React, { Component, useState } from 'react';
import axios from 'axios';
import Flashcard from './Flashcard';

function App() {
  //Variables for youtube link
  const [youtubeLink, setYoutubeLink] = useState("");
  const [keyConcepts, setKeyConcepts] = useState([]);

  const handleLinkChange = (event) => {
    setYoutubeLink(event.target.value);
  };

  const discardFlashcard = (index) => {
    setKeyConcepts(currentConcepts => currentConcepts.filter((_,i) => i !== index));
  }

  const sendLink = async () => {
    try {
      const response = await axios.post("http://localhost:8000/analyze_video", {
        youtube_link: youtubeLink,
      });

      const data = response.data;
      //Checks if the the response data is an array and if it exists
      if(data.key_concepts && Array.isArray(data.key_concepts)){
        console.log("Array has been found")
        setKeyConcepts(data.key_concepts);
      }
      else{
        console.error("Data does not contain key concepts: ", data);
        setKeyConcepts([]);
      }
    }catch(error){
      console.log(error);
      setKeyConcepts([]);
    }
  };

  return (
    <div className="App">
      <h1>Youtube Link to Flashcards Generator</h1>
      <div className="inputContainer">
        <input
          type="text"
          placeholder="Paste Youtube Link Here"
          value={youtubeLink}
          onChange={handleLinkChange}
          />
          <button onClick={sendLink}> Generate Flashcards
          </button>
      </div>

      <div className="flashcardsContainer">
        {keyConcepts.map((concept, index) =>(
          <Flashcard
            key={index}
            term={concept.term}
            definition={concept.definition}
            onDiscard={() =>discardFlashcard(index)}/>

        ))}
      </div>
    </div>
  )
}

export default App;