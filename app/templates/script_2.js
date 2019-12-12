 function startDictation() {

    if (window.hasOwnProperty('webkitSpeechRecognition')) {

      var recognition = new webkitSpeechRecognition();

      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.lang = "en-US";
      recognition.start();


      recognition.onresult = function(e) {
        var current = e.resultIndex;
        
  // Get a transcript of what was said.
        var transcript = ''
      var mobileRepeatBug = (transcript == e.results.transcript)//(document.getElementById('transcript').value == e.results[current].transcript); //[current]

      if(!mobileRepeatBug) {
        transcript += e.results.transcript
        document.getElementById('transcript').value = transcript;
        //document.getElementById('transcript').value = e.results[current].transcript; //[current
        recognition.stop()
        //transcript = e.results[current].transcript
                                        
      }        

        document.getElementById('transcript').value = transcript;
        
        //recognition.start(); 
      };



        /*recognition.onresult = function(e) {
        document.getElementById('transcript').value = e.results[0][0].transcript;
                                 
        recognition.stop();
        //document.getElementById('labnol').submit();
      };*/
     

    }
  }
